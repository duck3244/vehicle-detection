import { useMemo, useState } from 'react';
import { Loader2 } from 'lucide-react';
import type { DetectOptions, DetectResponse } from '@/api/client';
import { useDetectBatch, useJob } from '@/api/hooks';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { DEFAULT_OPTIONS, DetectionOptions } from '@/components/DetectionOptions';
import { Dropzone } from '@/components/Dropzone';

const MAX_BATCH = 10;

export function BatchPage() {
  const [files, setFiles] = useState<File[]>([]);
  const [jobId, setJobId] = useState<string | null>(null);
  const [options, setOptions] = useState<Required<DetectOptions>>(DEFAULT_OPTIONS);
  const [selectedResult, setSelectedResult] = useState<DetectResponse | null>(null);

  const submit = useDetectBatch();
  const job = useJob(jobId);

  const handleFiles = (incoming: File[]) => {
    const trimmed = incoming.slice(0, MAX_BATCH);
    setFiles(trimmed);
    setJobId(null);
    setSelectedResult(null);
    submit.reset();
  };

  const runBatch = () => {
    if (files.length === 0) return;
    submit.mutate(
      { files, options },
      {
        onSuccess: (res) => setJobId(res.job_id),
      },
    );
  };

  const progressPct = useMemo(() => {
    if (!job.data || job.data.total === 0) return 0;
    return Math.round((job.data.done / job.data.total) * 100);
  }, [job.data]);

  const pending = submit.isPending || (job.data && job.data.status !== 'done' && job.data.status !== 'failed');

  return (
    <div className="grid gap-6 lg:grid-cols-[360px_1fr]">
      <div className="space-y-4">
        <Card>
          <CardHeader>
            <CardTitle>이미지 업로드</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <Dropzone
              multiple
              maxFiles={MAX_BATCH}
              hint={`JPG/PNG/BMP/TIFF/WEBP · 최대 ${MAX_BATCH}장 · 각 10MB`}
              onFiles={handleFiles}
              disabled={Boolean(pending)}
            />
            {files.length > 0 && (
              <ul className="max-h-48 overflow-y-auto space-y-1 text-xs text-muted-foreground">
                {files.map((f) => (
                  <li key={f.name} className="flex justify-between gap-2">
                    <span className="truncate">{f.name}</span>
                    <span className="tabular-nums">{(f.size / 1024).toFixed(0)} KB</span>
                  </li>
                ))}
              </ul>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>옵션</CardTitle>
          </CardHeader>
          <CardContent>
            <DetectionOptions value={options} onChange={setOptions} disabled={Boolean(pending)} />
          </CardContent>
        </Card>

        <Button
          className="w-full"
          size="lg"
          onClick={runBatch}
          disabled={files.length === 0 || Boolean(pending)}
        >
          {pending && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
          {pending ? '처리 중...' : `배치 검출 (${files.length}장)`}
        </Button>

        {submit.isError && (
          <p className="text-sm text-destructive">제출 오류: {(submit.error as Error).message}</p>
        )}
      </div>

      <div className="space-y-4">
        {jobId && job.data && (
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle>진행 상황</CardTitle>
                <span className="text-xs text-muted-foreground">
                  {job.data.done}/{job.data.total} · {job.data.status}
                </span>
              </div>
            </CardHeader>
            <CardContent>
              <Progress value={progressPct} />
              {job.data.status === 'failed' && job.data.error && (
                <p className="mt-2 text-sm text-destructive">{job.data.error}</p>
              )}
            </CardContent>
          </Card>
        )}

        {job.data?.results && job.data.results.length > 0 && (
          <Card>
            <CardHeader>
              <CardTitle>결과 ({job.data.results.length}건)</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="overflow-hidden rounded-md border">
                <table className="w-full text-sm">
                  <thead className="bg-muted/50 text-xs uppercase text-muted-foreground">
                    <tr>
                      <th className="px-3 py-2 text-left">#</th>
                      <th className="px-3 py-2 text-left">Run ID</th>
                      <th className="px-3 py-2 text-right">검출</th>
                      <th className="px-3 py-2 text-right">추론 (ms)</th>
                      <th className="px-3 py-2 text-left">결과</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y">
                    {job.data.results.map((r, i) => (
                      <tr key={r.run_id} className="hover:bg-accent/40">
                        <td className="px-3 py-2 tabular-nums">{i + 1}</td>
                        <td className="px-3 py-2 font-mono text-xs">{r.run_id.slice(0, 8)}</td>
                        <td className="px-3 py-2 text-right tabular-nums">
                          {r.meta.num_detections}
                        </td>
                        <td className="px-3 py-2 text-right tabular-nums">
                          {r.meta.inference_ms.toFixed(0)}
                        </td>
                        <td className="px-3 py-2">
                          <div className="flex gap-2">
                            <button
                              type="button"
                              className="text-xs text-primary underline-offset-4 hover:underline"
                              onClick={() => setSelectedResult(r)}
                            >
                              미리보기
                            </button>
                            <a
                              href={r.annotated_image_url}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="text-xs text-primary underline-offset-4 hover:underline"
                            >
                              새 탭
                            </a>
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>
        )}

        {selectedResult && (
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle>미리보기</CardTitle>
                <button
                  type="button"
                  onClick={() => setSelectedResult(null)}
                  className="text-xs text-muted-foreground hover:text-foreground"
                >
                  닫기
                </button>
              </div>
            </CardHeader>
            <CardContent>
              <img
                src={selectedResult.annotated_image_url}
                alt="annotated"
                className="w-full rounded-md border"
              />
              <p className="mt-2 text-xs text-muted-foreground">
                검출 {selectedResult.meta.num_detections}건 · {selectedResult.meta.inference_ms.toFixed(0)}ms
              </p>
            </CardContent>
          </Card>
        )}

        {!jobId && (
          <Card>
            <CardContent className="py-16 text-center text-sm text-muted-foreground">
              이미지를 업로드하고 실행 버튼을 눌러 배치 검출을 시작하세요.
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
}
