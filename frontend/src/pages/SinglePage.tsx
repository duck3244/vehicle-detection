import { useEffect, useMemo, useState } from 'react';
import { Loader2 } from 'lucide-react';
import type { DetectOptions } from '@/api/client';
import { useDetectSingle } from '@/api/hooks';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Checkbox } from '@/components/ui/checkbox';
import { Label } from '@/components/ui/label';
import { DetectionCanvas } from '@/components/DetectionCanvas';
import { DetectionList } from '@/components/DetectionList';
import { DEFAULT_OPTIONS, DetectionOptions } from '@/components/DetectionOptions';
import { Dropzone } from '@/components/Dropzone';

export function SinglePage() {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [options, setOptions] = useState<Required<DetectOptions>>(DEFAULT_OPTIONS);
  const [showBbox, setShowBbox] = useState(true);
  const [showMask, setShowMask] = useState(true);
  const [showAnnotated, setShowAnnotated] = useState(false);
  const [selected, setSelected] = useState<number | null>(null);

  const mutation = useDetectSingle();

  useEffect(() => {
    if (!file) {
      setPreviewUrl(null);
      return;
    }
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  const handleFiles = (files: File[]) => {
    setFile(files[0] ?? null);
    mutation.reset();
    setSelected(null);
  };

  const runDetect = () => {
    if (!file) return;
    setSelected(null);
    mutation.mutate({ file, options });
  };

  const imageForCanvas = useMemo(() => {
    if (showAnnotated && mutation.data) return mutation.data.annotated_image_url;
    return previewUrl;
  }, [showAnnotated, mutation.data, previewUrl]);

  return (
    <div className="grid gap-6 lg:grid-cols-[360px_1fr]">
      <div className="space-y-4">
        <Card>
          <CardHeader>
            <CardTitle>이미지 업로드</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <Dropzone onFiles={handleFiles} disabled={mutation.isPending} />
            {file && (
              <p className="truncate text-xs text-muted-foreground" title={file.name}>
                선택됨: {file.name} ({(file.size / 1024).toFixed(0)} KB)
              </p>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>옵션</CardTitle>
          </CardHeader>
          <CardContent>
            <DetectionOptions value={options} onChange={setOptions} disabled={mutation.isPending} />
          </CardContent>
        </Card>

        <Button
          className="w-full"
          size="lg"
          onClick={runDetect}
          disabled={!file || mutation.isPending}
        >
          {mutation.isPending && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
          {mutation.isPending ? '검출 중...' : '검출 실행'}
        </Button>

        {mutation.isError && (
          <p className="text-sm text-destructive">
            오류: {(mutation.error as Error).message}
          </p>
        )}
      </div>

      <div className="space-y-4">
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle>결과</CardTitle>
              {mutation.data && (
                <div className="flex items-center gap-4 text-xs text-muted-foreground">
                  <span>{mutation.data.meta.inference_ms.toFixed(0)} ms</span>
                  <span>검출 {mutation.data.meta.num_detections}건</span>
                  <span>{mutation.data.meta.sam_used ? 'SAM on' : 'SAM off'}</span>
                </div>
              )}
            </div>
          </CardHeader>
          <CardContent className="space-y-4">
            {imageForCanvas ? (
              <>
                <div className="flex flex-wrap items-center gap-4 text-sm">
                  <label className="flex items-center gap-2">
                    <Checkbox
                      checked={showBbox}
                      onCheckedChange={(v) => setShowBbox(Boolean(v))}
                      disabled={showAnnotated}
                    />
                    <span>Bounding Box</span>
                  </label>
                  <label className="flex items-center gap-2">
                    <Checkbox
                      checked={showMask}
                      onCheckedChange={(v) => setShowMask(Boolean(v))}
                      disabled={showAnnotated || !mutation.data?.meta.sam_used}
                    />
                    <span>Mask</span>
                  </label>
                  {mutation.data && (
                    <label className="ml-auto flex items-center gap-2">
                      <Checkbox
                        checked={showAnnotated}
                        onCheckedChange={(v) => setShowAnnotated(Boolean(v))}
                      />
                      <Label>서버 주석 이미지</Label>
                    </label>
                  )}
                </div>
                <DetectionCanvas
                  imageUrl={imageForCanvas}
                  detections={showAnnotated ? [] : mutation.data?.detections ?? []}
                  showBbox={showBbox}
                  showMask={showMask}
                  selectedIndex={selected}
                />
              </>
            ) : (
              <p className="py-16 text-center text-sm text-muted-foreground">
                이미지를 업로드하면 여기에 표시됩니다.
              </p>
            )}
          </CardContent>
        </Card>

        {mutation.data && (
          <Card>
            <CardHeader>
              <CardTitle>검출 목록</CardTitle>
            </CardHeader>
            <CardContent>
              <DetectionList
                detections={mutation.data.detections}
                selectedIndex={selected}
                onSelect={setSelected}
              />
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
}
