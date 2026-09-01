import React from 'react';
import { Folder, FileCode, FileText, ChevronRight, ChevronDown } from 'lucide-react';

export default function FileExplorer({ files }) {
  // files is a nested object/array representing the file tree
  if (!files || files.length === 0) {
    return (
      <div className="text-gray-500 text-sm italic p-4">
        No files indexed yet.
      </div>
    );
  }

  return (
    <div className="overflow-y-auto h-full p-2 space-y-1 custom-scrollbar">
      {files.map((node, i) => (
        <FileNode key={i} node={node} level={0} />
      ))}
    </div>
  );
}

function FileNode({ node, level }) {
  const [isOpen, setIsOpen] = React.useState(level < 1);
  const isDir = node.type === 'directory';

  return (
    <div className="w-full">
      <div
        className="flex items-center gap-1.5 py-1 px-2 rounded-md hover:bg-gray-800 cursor-pointer text-sm text-gray-300 transition-colors"
        style={{ paddingLeft: `${level * 12 + 8}px` }}
        onClick={() => isDir && setIsOpen(!isOpen)}
      >
        {isDir ? (
          isOpen ? <ChevronDown size={14} className="text-gray-500" /> : <ChevronRight size={14} className="text-gray-500" />
        ) : (
          <span className="w-3.5 inline-block" /> // alignment spacer
        )}
        
        {isDir ? (
          <Folder size={16} className="text-blue-400" />
        ) : (
          node.name.endsWith('.js') || node.name.endsWith('.py') ? (
            <FileCode size={16} className="text-yellow-400" />
          ) : (
            <FileText size={16} className="text-gray-400" />
          )
        )}
        <span className="truncate flex-1 select-none">{node.name}</span>
      </div>
      
      {isDir && isOpen && node.children && (
        <div className="flex flex-col w-full">
          {node.children.map((child, i) => (
            <FileNode key={i} node={child} level={level + 1} />
          ))}
        </div>
      )}
    </div>
  );
}
