import { useDropzone } from 'react-dropzone';
import { Upload } from 'lucide-react';
import { cn } from '@/lib/utils';

const ACCEPT = {
  'image/jpeg': ['.jpg', '.jpeg'],
  'image/png': ['.png'],
  'image/bmp': ['.bmp'],
  'image/tiff': ['.tif', '.tiff'],
  'image/webp': ['.webp'],
};

interface DropzoneProps {
  onFiles: (files: File[]) => void;
  multiple?: boolean;
  maxFiles?: number;
  maxSize?: number;
  disabled?: boolean;
  className?: string;
  hint?: string;
}

export function Dropzone({
  onFiles,
  multiple = false,
  maxFiles,
  maxSize = 10 * 1024 * 1024,
  disabled,
  className,
  hint,
}: DropzoneProps) {
  const { getRootProps, getInputProps, isDragActive, fileRejections } = useDropzone({
    accept: ACCEPT,
    multiple,
    maxFiles,
    maxSize,
    disabled,
    onDrop: (accepted) => {
      if (accepted.length > 0) onFiles(accepted);
    },
  });

  return (
    <div className={className}>
      <div
        {...getRootProps()}
        className={cn(
          'flex flex-col items-center justify-center rounded-lg border-2 border-dashed border-muted-foreground/30 bg-muted/30 px-6 py-10 text-center transition-colors',
          isDragActive && 'border-primary bg-accent',
          disabled && 'pointer-events-none opacity-60',
        )}
      >
        <input {...getInputProps()} />
        <Upload className="mb-2 h-6 w-6 text-muted-foreground" />
        <p className="text-sm font-medium">
          {isDragActive ? '여기에 놓으세요' : '이미지를 드롭하거나 클릭해 선택'}
        </p>
        <p className="mt-1 text-xs text-muted-foreground">
          {hint ?? 'JPG/PNG/BMP/TIFF/WEBP · 최대 10MB'}
        </p>
      </div>
      {fileRejections.length > 0 && (
        <ul className="mt-2 text-xs text-destructive">
          {fileRejections.map(({ file, errors }) => (
            <li key={file.name}>
              {file.name}: {errors.map((e) => e.message).join(', ')}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
