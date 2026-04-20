import { useMutation, useQuery } from '@tanstack/react-query';
import {
  detectBatch,
  detectSingle,
  getHealth,
  getJob,
  getModels,
  type DetectOptions,
} from './client';

export function useHealth() {
  return useQuery({ queryKey: ['health'], queryFn: getHealth, refetchInterval: 30_000 });
}

export function useModels() {
  return useQuery({ queryKey: ['models'], queryFn: getModels, staleTime: Infinity });
}

export function useDetectSingle() {
  return useMutation({
    mutationFn: ({ file, options }: { file: File; options?: DetectOptions }) =>
      detectSingle(file, options ?? {}),
  });
}

export function useDetectBatch() {
  return useMutation({
    mutationFn: ({ files, options }: { files: File[]; options?: DetectOptions }) =>
      detectBatch(files, options ?? {}),
  });
}

export function useJob(jobId: string | null) {
  return useQuery({
    queryKey: ['job', jobId],
    queryFn: () => getJob(jobId!),
    enabled: Boolean(jobId),
    refetchInterval: (query) => {
      const data = query.state.data;
      if (!data) return 1000;
      return data.status === 'done' || data.status === 'failed' ? false : 1000;
    },
  });
}
