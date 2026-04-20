import type { DetectOptions } from '@/api/client';
import { Checkbox } from '@/components/ui/checkbox';
import { Label } from '@/components/ui/label';
import { Slider } from '@/components/ui/slider';

interface DetectionOptionsProps {
  value: Required<DetectOptions>;
  onChange: (next: Required<DetectOptions>) => void;
  disabled?: boolean;
}

export const DEFAULT_OPTIONS: Required<DetectOptions> = {
  useSam: true,
  confidence: 0.25,
};

export function DetectionOptions({ value, onChange, disabled }: DetectionOptionsProps) {
  const update = (patch: Partial<Required<DetectOptions>>) => onChange({ ...value, ...patch });

  return (
    <div className="space-y-5">
      <div className="flex items-center gap-2">
        <Checkbox
          id="use-sam"
          checked={value.useSam}
          onCheckedChange={(v) => update({ useSam: Boolean(v) })}
          disabled={disabled}
        />
        <Label htmlFor="use-sam">SAM 세그멘테이션 사용</Label>
      </div>

      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <Label>신뢰도 임계값</Label>
          <span className="text-sm tabular-nums text-muted-foreground">
            {value.confidence.toFixed(2)}
          </span>
        </div>
        <Slider
          min={0.05}
          max={0.95}
          step={0.05}
          value={[value.confidence]}
          onValueChange={([v]) => update({ confidence: v })}
          disabled={disabled}
        />
      </div>
    </div>
  );
}
