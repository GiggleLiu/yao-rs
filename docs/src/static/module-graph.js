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
      var moduleData = {};

      data.modules.forEach(function(mod) {
        var nodeId = 'mod_' + mod.name;
        moduleData[nodeId] = mod;
        elements.push({
          data: {
            id: nodeId,
            label: mod.name,
            category: mod.category,
            doc_path: mod.doc_path,
            itemCount: mod.items.length
          },
          position: fixedPositions[mod.name] || { x: 500, y: 280 }
        });
      });

      data.edges.forEach(function(e) {
        elements.push({
          data: {
            id: 'edge_' + e.source + '_' + e.target,
            source: 'mod_' + e.source,
            target: 'mod_' + e.target
          }
        });
      });

      var cy = cytoscape({
        container: container,
        elements: elements,
        style: [
          { selector: 'node', style: {
            'label': 'data(label)',
            'text-valign': 'center', 'text-halign': 'center',
            'font-size': '12px', 'font-family': 'monospace', 'font-weight': 'bold',
            'width': function(ele) { return Math.max(ele.data('label').length * 8 + 20, 80); },
            'height': 36,
            'shape': 'round-rectangle',
            'background-color': function(ele) { return categoryColors[ele.data('category')] || '#f0f0f0'; },
            'border-width': 2,
            'border-color': function(ele) { return categoryBorders[ele.data('category')] || '#999'; },
            'cursor': 'pointer'
          }},
          { selector: 'node.selected-mod', style: {
            'border-width': 3,
            'border-color': '#2196F3'
          }},
          { selector: 'edge', style: {
            'width': 1.5, 'line-color': '#999', 'target-arrow-color': '#999',
            'target-arrow-shape': 'triangle', 'curve-style': 'bezier',
            'arrow-scale': 0.8,
            'source-distance-from-node': 5,
            'target-distance-from-node': 5
          }},
          { selector: '.faded', style: { 'opacity': 0.15 } }
        ],
        layout: { name: 'preset' },
        userZoomingEnabled: true,
        userPanningEnabled: true,
        boxSelectionEnabled: false
      });

      cy.fit(40);

      // Detail panel: show public items on click
      var detail = document.getElementById('mg-detail');

      function showDetail(mod) {
        if (!detail) return;
        var html = '<strong>' + mod.name + '</strong> <span class="mg-detail-count">(' + mod.items.length + ' public items)</span>';
        html += '<div class="mg-detail-items">';
        mod.items.forEach(function(item) {
          var icon = kindIcons[item.kind] || item.kind;
          html += '<span class="mg-detail-item"><code>' + icon + '</code> ' + item.name + '</span>';
        });
        html += '</div>';
        detail.innerHTML = html;
        detail.style.display = 'block';
      }

      // Tooltip
      var tooltip = document.getElementById('mg-tooltip');
      cy.on('mouseover', 'node', function(evt) {
        var d = evt.target.data();
        tooltip.innerHTML = '<strong>' + d.label + '</strong> (' + d.itemCount + ' items)';
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

      // Click node: show public items
      cy.on('tap', 'node', function(evt) {
        cy.nodes().removeClass('selected-mod');
        evt.target.addClass('selected-mod');
        var mod = moduleData[evt.target.id()];
        if (mod) showDetail(mod);
      });

      // Double-click: open rustdoc
      cy.on('dbltap', 'node', function(evt) {
        var d = evt.target.data();
        if (d.doc_path) {
          window.open('rustdoc/yao_rs/' + d.doc_path, '_blank');
        }
      });

      // Click background: clear
      cy.on('tap', function(evt) {
        if (evt.target === cy) {
          cy.nodes().removeClass('selected-mod');
          if (detail) detail.style.display = 'none';
        }
      });
    })
    .catch(function(err) {
      container.innerHTML = '<p style="padding:1em;color:#c00;">Failed to load module graph: ' + err.message + '</p>';
    });
});
