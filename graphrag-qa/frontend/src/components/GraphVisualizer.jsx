import React, { useMemo } from 'react';
import { ReactFlow, Controls, Background, MarkerType } from '@xyflow/react';
import '@xyflow/react/dist/style.css';

export default function GraphVisualizer({ retrievalGraph }) {
  const initialNodes = useMemo(() => {
    if (!retrievalGraph || !retrievalGraph.nodes) return [];
    
    const numNodes = retrievalGraph.nodes.length;
    const radius = Math.max(150, numNodes * 20); // Dynamic radius
    const centerX = 250;
    const centerY = 250;
    
    return retrievalGraph.nodes.map((node, i) => {
      // Circular layout
      const angle = (i / numNodes) * 2 * Math.PI;
      const x = centerX + radius * Math.cos(angle);
      const y = centerY + radius * Math.sin(angle);
      
      // Shorten label (e.g. src/flask/app.py:Flask.dispatch_request:78 -> Flask.dispatch_request)
      let label = node.label || node.id;
      const parts = label.split(':');
      if (parts.length >= 2) {
        label = parts[parts.length - 2];
      }

      return {
        id: node.id,
        position: { x, y },
        data: { label: label },
        type: 'default',
        style: {
          background: node.is_active ? '#2563eb' : '#1f2937',
          color: 'white',
          border: '1px solid #374151',
          borderRadius: '8px',
          fontSize: '11px',
          padding: '8px 12px',
          width: 'max-content',
          maxWidth: '180px',
          textAlign: 'center',
          boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
        }
      };
    });
  }, [retrievalGraph]);

  const initialEdges = useMemo(() => {
    if (!retrievalGraph || !retrievalGraph.edges) return [];
    return retrievalGraph.edges.map((edge, i) => ({
      id: `e-${edge.source}-${edge.target}`,
      source: edge.source,
      target: edge.target,
      animated: edge.source !== edge.target, // animate if not self-referencing
      style: { stroke: '#4b5563', strokeWidth: 1.5 },
      markerEnd: { type: MarkerType.ArrowClosed, color: '#4b5563' },
    }));
  }, [retrievalGraph]);

  if (!retrievalGraph || !retrievalGraph.nodes || retrievalGraph.nodes.length === 0) {
    return (
      <div className="flex items-center justify-center h-full text-gray-500 text-sm">
        Waiting for graph retrieval...
      </div>
    );
  }

  return (
    <div className="h-full w-full bg-gray-950 relative">
      <ReactFlow
        nodes={initialNodes}
        edges={initialEdges}
        fitView
        fitViewOptions={{ padding: 0.2 }}
        className="dark-theme"
      >
        <Background color="#374151" gap={16} />
        <Controls className="!bg-gray-800 !border-gray-700 !fill-white" />
      </ReactFlow>
    </div>
  );
}
