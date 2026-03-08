document.addEventListener('DOMContentLoaded', function() {
  var container = document.getElementById('module-graph');
  if (!container) return;

  var categoryColors = {
    core: '#c8f0c8', simulation: '#c8c8f0', tensor: '#f0f0a0',
    utility: '#e0e0e0', higher: '#e0c8f0', visualization: '#f0c8e0'
  };
  var categoryBorders = {
    core: '#4a8c4a', simulation: '#4a4a8c', tensor: '#8c8c4a',
    utility: '#888888', higher: '#6a4a8c', visualization: '#8c4a6a'
  };
  var kindIcons = {
    'struct': 'S', 'enum': 'E', 'function': 'fn', 'trait': 'T',
    'type_alias': 'type', 'constant': 'const'
  };

  // Fixed positions grouped by category (columns)
  var fixedPositions = {
    // Core
    'gate':           { x: 100, y: 100 },
    'circuit':        { x: 100, y: 280 },
    'state':          { x: 100, y: 460 },
    // Simulation
    'apply':          { x: 310, y: 60 },
    'instruct':       { x: 310, y: 210 },
    'instruct_qubit': { x: 310, y: 360 },
    'measure':        { x: 310, y: 500 },
    // Tensor Export
    'einsum':         { x: 520, y: 190 },
    'tensors':        { x: 520, y: 370 },
    // Higher-level
    'easybuild':      { x: 720, y: 100 },
    'operator':       { x: 720, y: 280 },
    'noise':          { x: 720, y: 460 },
    // Utilities
    'index':          { x: 910, y: 100 },
    'bitutils':       { x: 910, y: 280 },
    'json':           { x: 910, y: 460 }
  };

  fetch('static/module-graph.json')
    .then(function(r) { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json(); })
    .then(function(data) {
      var elements = [];

      data.modules.forEach(function(mod) {
        var parentId = 'mod_' + mod.name;
        var pos = fixedPositions[mod.name] || { x: 500, y: 280 };

        // Compound parent node
        elements.push({
          data: {
            id: parentId,
            label: mod.name,
            category: mod.category,
            doc_path: mod.doc_path,
            itemCount: mod.items.length,
            isParent: true
          },
          position: pos
        });

        // Child item nodes (positioned in a vertical list below parent center)
        mod.items.forEach(function(item, idx) {
          var childId = mod.name + '::' + item.name;
          var icon = kindIcons[item.kind] || item.kind;
          elements.push({
            data: {
              id: childId,
              parent: parentId,
              label: icon + ' ' + item.name,
              fullLabel: mod.name + '::' + item.name,
              category: mod.category,
              kind: item.kind,
              doc: item.doc || '',
              isChild: true,
              moduleName: mod.name,
              itemName: item.name
            },
            position: {
              x: pos.x,
              y: pos.y + 18 + idx * 22
            }
          });
        });
      });

      // Module-level edges
      data.edges.forEach(function(e) {
        elements.push({
          data: {
            id: 'edge_' + e.source + '_' + e.target,
            source: 'mod_' + e.source,
            target: 'mod_' + e.target
          }
        });
      });

      if (!window.cytoscape) {
        container.innerHTML = '<p style="font-family: sans-serif; font-size: 14px; color: #555;">' +
          'Module graph visualization is unavailable because the required library failed to load.' +
          '</p>';
        return;
      }

      var cy = window.cytoscape({
        container: container,
        elements: elements,
        style: [
          // Module nodes (compound parents)
          { selector: 'node[?isParent]', style: {
            'label': 'data(label)',
            'text-valign': 'center', 'text-halign': 'center',
            'font-size': '12px', 'font-family': 'monospace', 'font-weight': 'bold',
            'min-width': function(ele) { return Math.max(ele.data('label').length * 8 + 20, 80); },
            'min-height': 36,
            'padding': '4px',
            'shape': 'round-rectangle',
            'background-color': function(ele) { return categoryColors[ele.data('category')] || '#f0f0f0'; },
            'border-width': 2,
            'border-color': function(ele) { return categoryBorders[ele.data('category')] || '#999'; },
            'compound-sizing-wrt-labels': 'include',
            'cursor': 'pointer'
          }},
          // Expanded parent
          { selector: 'node[?isParent].expanded', style: {
            'text-valign': 'top',
            'padding': '10px'
          }},
          // Child item nodes
          { selector: 'node[?isChild]', style: {
            'label': 'data(label)',
            'text-valign': 'center', 'text-halign': 'center',
            'font-size': '9px', 'font-family': 'monospace',
            'width': function(ele) { return Math.max(ele.data('label').length * 5.5 + 8, 40); },
            'height': 18,
            'shape': 'round-rectangle',
            'background-color': function(ele) { return categoryColors[ele.data('category')] || '#f0f0f0'; },
            'border-width': 1,
            'border-color': function(ele) { return categoryBorders[ele.data('category')] || '#999'; }
          }},
          // Edges
          { selector: 'edge', style: {
            'width': 1.5, 'line-color': '#999', 'target-arrow-color': '#999',
            'target-arrow-shape': 'triangle', 'curve-style': 'bezier',
            'arrow-scale': 0.8,
            'source-distance-from-node': 5,
            'target-distance-from-node': 5
          }}
        ],
        layout: { name: 'preset' },
        userZoomingEnabled: true,
        userPanningEnabled: true,
        boxSelectionEnabled: false
      });

      // Initial state: hide all children, then position parents at fixed positions
      cy.nodes('[?isChild]').style('display', 'none');
      Object.keys(fixedPositions).forEach(function(name) {
        var node = cy.getElementById('mod_' + name);
        if (node.length) node.position(fixedPositions[name]);
      });
      cy.fit(40);

      var expandedParents = {};

      // Click: toggle expand/collapse
      cy.on('tap', 'node[?isParent]', function(evt) {
        var parentNode = evt.target;
        var parentId = parentNode.id();
        var children = parentNode.children();

        if (expandedParents[parentId]) {
          // Collapse
          children.style('display', 'none');
          parentNode.removeClass('expanded');
          expandedParents[parentId] = false;
          var name = parentNode.data('label');
          if (fixedPositions[name]) {
            parentNode.position(fixedPositions[name]);
          }
        } else {
          // Expand
          children.style('display', 'element');
          parentNode.addClass('expanded');
          expandedParents[parentId] = true;
        }
      });

      // Rustdoc URL prefixes by kind
      var kindPrefix = {
        'function': 'fn', 'struct': 'struct', 'enum': 'enum',
        'trait': 'trait', 'type_alias': 'type', 'constant': 'constant'
      };

      // Double-click: open rustdoc
      cy.on('dbltap', 'node[?isParent]', function(evt) {
        var d = evt.target.data();
        if (d.doc_path) {
          window.open('api/yao_rs/' + d.doc_path, '_blank');
        }
      });
      cy.on('dbltap', 'node[?isChild]', function(evt) {
        var d = evt.target.data();
        var prefix = kindPrefix[d.kind] || d.kind;
        window.open('api/yao_rs/' + d.moduleName + '/' + prefix + '.' + d.itemName + '.html', '_blank');
      });

      // Tooltip
      var tooltip = document.getElementById('mg-tooltip');
      function escapeHtml(s) {
        var div = document.createElement('div');
        div.appendChild(document.createTextNode(s));
        return div.innerHTML;
      }
      cy.on('mouseover', 'node[?isParent]', function(evt) {
        var d = evt.target.data();
        tooltip.innerHTML = '<strong>' + escapeHtml(d.label) + '</strong> (' + d.itemCount + ' items)';
        tooltip.style.display = 'block';
      });
      cy.on('mouseover', 'node[?isChild]', function(evt) {
        var d = evt.target.data();
        var html = '<strong>' + escapeHtml(d.fullLabel) + '</strong><br><code>' + escapeHtml(d.kind) + '</code>';
        if (d.doc) html += '<br><em>' + escapeHtml(d.doc) + '</em>';
        tooltip.innerHTML = html;
        tooltip.style.display = 'block';
      });
      cy.on('mousemove', 'node', function(evt) {
        var pos = evt.renderedPosition || evt.position;
        var rect = container.getBoundingClientRect();
        tooltip.style.left = (rect.left + window.scrollX + pos.x + 15) + 'px';
        tooltip.style.top = (rect.top + window.scrollY + pos.y - 10) + 'px';
      });
      cy.on('mouseout', 'node', function() { tooltip.style.display = 'none'; });

      // Edge tooltip
      cy.on('mouseover', 'edge', function(evt) {
        var src = evt.target.source().data('label');
        var dst = evt.target.target().data('label');
        tooltip.innerHTML = '<strong>' + src + ' \u2192 ' + dst + '</strong>';
        tooltip.style.display = 'block';
      });
      cy.on('mousemove', 'edge', function(evt) {
        var pos = evt.renderedPosition || evt.position;
        var rect = container.getBoundingClientRect();
        tooltip.style.left = (rect.left + window.scrollX + pos.x + 15) + 'px';
        tooltip.style.top = (rect.top + window.scrollY + pos.y - 10) + 'px';
      });
      cy.on('mouseout', 'edge', function() { tooltip.style.display = 'none'; });
    })
    .catch(function(err) {
      container.innerHTML = '<p style="padding:1em;color:#c00;">Failed to load module graph: ' + err.message + '</p>';
    });
});
