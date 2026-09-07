use super::*;
use crate::contraction_plan::{optimize_code, tensor_shape, tensor_shapes, validate_tree};
use omeco::{EinCode, GreedyMethod, Label, NestedEinsum};
use std::collections::{HashMap, HashSet};
use tenferro_einsum::{EagerEinsumExt, EinsumSubscripts};

/// Fixed shapes and contraction order, independent of tensor values and devices.
/// Each explicit node is retained. N-ary nodes execute left to right.
pub struct PreparedCudaContraction {
    input_shapes: Vec<Vec<usize>>,
    output_shape: Vec<usize>,
    root: Node,
    has_diagonals: bool,
}

enum Node {
    Input(usize),
    Contract {
        children: Vec<Node>,
        expressions: Vec<EinsumSubscripts>,
    },
}

impl PreparedCudaContraction {
    /// Validate shapes and an explicit tree, or use deterministic omeco greedy
    /// planning. No CUDA initialization or tensor-value allocation occurs here.
    pub fn new<L: Label>(
        code: &EinCode<L>,
        sizes: &HashMap<L, usize>,
        tree: Option<&NestedEinsum<L>>,
    ) -> Result<Self, String> {
        let shapes = tensor_shapes(code, sizes)?;
        let generated;
        let tree = match tree {
            Some(tree) => tree,
            None => {
                generated = optimize_code(code, sizes, &GreedyMethod::default())
                    .ok_or("Greedy contraction planning failed")?;
                &generated
            }
        };
        validate_tree(tree, code)?;
        let mut labels = HashMap::new();
        for label in code.ixs.iter().flatten() {
            let next = u32::try_from(labels.len()).map_err(error)?;
            labels.entry(label.clone()).or_insert(next);
        }
        Ok(Self {
            input_shapes: shapes.inputs,
            output_shape: shapes.output,
            root: lower(tree, &labels, sizes)?,
            has_diagonals: code.ixs.iter().any(|xs| {
                let mut seen = HashSet::new();
                xs.iter().any(|x| !seen.insert(x))
            }),
        })
    }

    /// Required complex128 input shapes, in original network input order.
    pub fn input_shapes(&self) -> &[Vec<usize>] {
        &self.input_shapes
    }
    /// Result axes follow the exact requested output order.
    pub fn output_shape(&self) -> &[usize] {
        &self.output_shape
    }
}

fn lower<L: Label>(
    tree: &NestedEinsum<L>,
    labels: &HashMap<L, u32>,
    sizes: &HashMap<L, usize>,
) -> Result<Node, String> {
    match tree {
        NestedEinsum::Leaf { tensor_index } => Ok(Node::Input(*tensor_index)),
        NestedEinsum::Node { args, eins } => {
            tensor_shape(&eins.iy, sizes)?;
            let children = args
                .iter()
                .map(|arg| lower(arg, labels, sizes))
                .collect::<Result<_, _>>()?;
            let inputs: Vec<Vec<u32>> = eins
                .ixs
                .iter()
                .map(|xs| xs.iter().map(|l| labels[l]).collect())
                .collect();
            let output: Vec<u32> = eins.iy.iter().map(|l| labels[l]).collect();
            let mut expressions = Vec::new();
            if inputs.len() == 1 {
                expressions.push(EinsumSubscripts { inputs, output });
            } else if let Some(first) = inputs.first() {
                let mut current = first.clone();
                for i in 1..inputs.len() {
                    let result = if i + 1 == inputs.len() {
                        output.clone()
                    } else {
                        let needed: HashSet<_> = output
                            .iter()
                            .chain(inputs[i + 1..].iter().flatten())
                            .copied()
                            .collect();
                        let mut seen = HashSet::new();
                        current
                            .iter()
                            .chain(&inputs[i])
                            .filter(|l| needed.contains(l) && seen.insert(**l))
                            .copied()
                            .collect()
                    };
                    expressions.push(EinsumSubscripts {
                        inputs: vec![current, inputs[i].clone()],
                        output: result.clone(),
                    });
                    current = result;
                }
            }
            Ok(Node::Contract {
                children,
                expressions,
            })
        }
    }
}

impl CudaSimulator {
    /// Contract already-uploaded complex128 tensors, retaining GPU residency.
    /// Values may change between runs while shapes, order and runtime must match.
    /// The result composes with eager AD when any tensor input tracks gradients.
    /// Repeated labels within an input (trace/diagonal extraction) support forward
    /// execution only with tenferro 0.4.0; tracked inputs are rejected in that case.
    pub fn contract(
        &self,
        plan: &PreparedCudaContraction,
        inputs: &[EagerTensor],
    ) -> Result<EagerTensor, String> {
        if std::env::var_os("TENFERRO_EAGER_WHOLE_PROGRAM").is_some() {
            return Err("Unset TENFERRO_EAGER_WHOLE_PROGRAM for CUDA contraction: tenferro 0.4.0's prototype executor contains host-only operations".into());
        }
        if inputs.len() != plan.input_shapes.len() {
            return Err("CUDA contraction input count mismatch".into());
        }
        for (tensor, shape) in inputs.iter().zip(&plan.input_shapes) {
            validate_context(&self.runtime, tensor)?;
            if tensor.dtype() != DType::C64 || tensor.shape() != shape {
                return Err("CUDA contraction input shape or dtype mismatch".into());
            }
        }
        if plan.has_diagonals && inputs.iter().any(EagerTensor::tracks_grad) {
            return Err("CUDA trace/diagonal gradients are unsupported by tenferro 0.4.0; use untracked tensors for forward contraction".into());
        }
        let result = execute(self, &plan.root, inputs)?;
        validate_context(&self.runtime, &result)?;
        if result.shape() != plan.output_shape {
            return Err("CUDA contraction returned an unexpected shape".into());
        }
        Ok(result)
    }
}

fn execute(
    gpu: &CudaSimulator,
    node: &Node,
    inputs: &[EagerTensor],
) -> Result<EagerTensor, String> {
    match node {
        Node::Input(i) => Ok(inputs[*i].clone()),
        Node::Contract {
            children,
            expressions,
        } => {
            let mut children = children.iter();
            let Some(first) = children.next() else {
                return gpu.upload(
                    &Tensor::from_vec_col_major(vec![], vec![C::new(1., 0.)]).map_err(error)?,
                    false,
                );
            };
            let mut value = execute(gpu, first, inputs)?;
            if children.len() == 0 {
                return unary(value, &expressions[0]);
            }
            for (child, expression) in children.zip(expressions) {
                let rhs = execute(gpu, child, inputs)?;
                value = [&value, &rhs]
                    .einsum_subscripts(expression)
                    .map_err(error)?;
            }
            Ok(value)
        }
    }
}

// The upstream unary eager extension uses a host-only duplication path.
// Use its standard device operations for traces, diagonals and permutations.
fn unary(mut tensor: EagerTensor, expression: &EinsumSubscripts) -> Result<EagerTensor, String> {
    let mut labels = expression.inputs[0].clone();
    loop {
        let duplicate = labels
            .iter()
            .enumerate()
            .find_map(|(b, label)| labels[..b].iter().position(|x| x == label).map(|a| (a, b)));
        let Some((a, b)) = duplicate else { break };
        tensor = tensor.extract_diag(a, b).map_err(error)?;
        labels.remove(b);
    }
    let axes: Vec<_> = labels
        .iter()
        .enumerate()
        .filter_map(|(i, l)| (!expression.output.contains(l)).then_some(i))
        .collect();
    if !axes.is_empty() {
        tensor = tensor.reduce_sum(Some(&axes)).map_err(error)?;
        labels.retain(|l| expression.output.contains(l));
    }
    let perm: Vec<_> = expression
        .output
        .iter()
        .map(|l| labels.iter().position(|x| x == l).unwrap())
        .collect();
    tensor.transpose(&perm).map_err(error)
}
