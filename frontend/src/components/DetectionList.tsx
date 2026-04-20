import { cn } from '@/lib/utils';
import type { DetectionItem } from '@/api/client';

interface DetectionListProps {
  detections: DetectionItem[];
  selectedIndex: number | null;
  onSelect: (index: number | null) => void;
}

export function DetectionList({ detections, selectedIndex, onSelect }: DetectionListProps) {
  if (detections.length === 0) {
    return (
      <p className="py-8 text-center text-sm text-muted-foreground">검출된 차량이 없습니다.</p>
    );
  }

  return (
    <div className="divide-y rounded-md border">
      {detections.map((d, i) => {
        const [x1, y1, x2, y2] = d.bbox;
        const isSelected = selectedIndex === i;
        return (
          <button
            key={i}
            type="button"
            onClick={() => onSelect(isSelected ? null : i)}
            className={cn(
              'flex w-full items-center justify-between gap-3 px-4 py-2 text-left text-sm hover:bg-accent',
              isSelected && 'bg-accent',
            )}
          >
            <div className="flex items-center gap-3">
              <span className="inline-flex h-6 w-6 items-center justify-center rounded-full bg-muted text-xs font-medium">
                {i + 1}
              </span>
              <span className="font-medium">{d.class_kr}</span>
              <span className="text-xs text-muted-foreground">({d.class})</span>
            </div>
            <div className="flex items-center gap-4 text-xs text-muted-foreground tabular-nums">
              <span>{(d.score * 100).toFixed(1)}%</span>
              <span>
                [{x1.toFixed(0)}, {y1.toFixed(0)}, {x2.toFixed(0)}, {y2.toFixed(0)}]
              </span>
            </div>
          </button>
        );
      })}
    </div>
  );
}
