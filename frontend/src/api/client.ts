import axios from 'axios';
import type { components } from './schema';

export const api = axios.create({
  baseURL: '/api',
  timeout: 90_000,
});

export type DetectResponse = components['schemas']['DetectResponse'];
export type DetectionItem = components['schemas']['DetectionItem'];
export type DetectMeta = components['schemas']['DetectMeta'];
export type BatchJobResponse = components['schemas']['BatchJobResponse'];
export type JobStatusResponse = components['schemas']['JobStatusResponse'];
export type ModelsResponse = components['schemas']['ModelsResponse'];
export type HealthResponse = components['schemas']['HealthResponse'];

export interface DetectOptions {
  useSam?: boolean;
  confidence?: number;
}

function appendOptions(form: FormData, opts: DetectOptions): void {
  if (opts.useSam !== undefined) form.append('use_sam', String(opts.useSam));
  if (opts.confidence !== undefined) form.append('confidence', String(opts.confidence));
}

export async function detectSingle(file: File, opts: DetectOptions = {}): Promise<DetectResponse> {
  const form = new FormData();
  form.append('file', file);
  appendOptions(form, opts);
  const { data } = await api.post<DetectResponse>('/detect', form);
  return data;
}

export async function detectBatch(files: File[], opts: DetectOptions = {}): Promise<BatchJobResponse> {
  const form = new FormData();
  files.forEach((f) => form.append('files', f));
  appendOptions(form, opts);
  const { data } = await api.post<BatchJobResponse>('/detect/batch', form);
  return data;
}

export async function getJob(jobId: string): Promise<JobStatusResponse> {
  const { data } = await api.get<JobStatusResponse>(`/jobs/${jobId}`);
  return data;
}

export async function getHealth(): Promise<HealthResponse> {
  const { data } = await api.get<HealthResponse>('/system/health');
  return data;
}

export async function getModels(): Promise<ModelsResponse> {
  const { data } = await api.get<ModelsResponse>('/models');
  return data;
}
