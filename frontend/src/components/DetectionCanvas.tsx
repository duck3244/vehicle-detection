import { Fragment, useEffect, useRef, useState } from 'react';
import { Image as KonvaImage, Layer, Rect, Stage, Text } from 'react-konva';
import type { DetectionItem } from '@/api/client';

interface DetectionCanvasProps {
  imageUrl: string;
  detections: DetectionItem[];
  showBbox: boolean;
  showMask: boolean;
  selectedIndex?: number | null;
}

const COLORS = ['#ef4444', '#3b82f6', '#22c55e', '#f59e0b', '#a855f7', '#06b6d4', '#ec4899'];

function useImage(src: string): HTMLImageElement | null {
  const [image, setImage] = useState<HTMLImageElement | null>(null);
  useEffect(() => {
    if (!src) {
      setImage(null);
      return;
    }
    const img = new window.Image();
    img.crossOrigin = 'anonymous';
    img.src = src;
    img.onload = () => setImage(img);
    img.onerror = () => setImage(null);
    return () => {
      img.onload = null;
      img.onerror = null;
    };
  }, [src]);
  return image;
}

function MaskImage({ url }: { url: string }) {
  const img = useImage(url);
  if (!img) return null;
  return (
    <KonvaImage
      image={img}
      width={img.naturalWidth}
      height={img.naturalHeight}
      opacity={0.4}
      listening={false}
    />
  );
}

export function DetectionCanvas({
  imageUrl,
  detections,
  showBbox,
  showMask,
  selectedIndex,
}: DetectionCanvasProps) {
  const image = useImage(imageUrl);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [containerWidth, setContainerWidth] = useState(0);

  useEffect(() => {
    if (!containerRef.current) return;
    const el = containerRef.current;
    const ro = new ResizeObserver(() => setContainerWidth(el.clientWidth));
    ro.observe(el);
    setContainerWidth(el.clientWidth);
    return () => ro.disconnect();
  }, []);

  if (!image) {
    return (
      <div
        ref={containerRef}
        className="flex h-96 w-full items-center justify-center rounded-md border bg-muted/40 text-sm text-muted-foreground"
      >
        이미지 로딩 중...
      </div>
    );
  }

  const scale = containerWidth ? Math.min(1, containerWidth / image.naturalWidth) : 1;
  const stageWidth = image.naturalWidth * scale;
  const stageHeight = image.naturalHeight * scale;

  return (
    <div ref={containerRef} className="w-full overflow-hidden rounded-md border bg-background">
      <Stage width={stageWidth} height={stageHeight} scale={{ x: scale, y: scale }}>
        <Layer listening={false}>
          <KonvaImage image={image} width={image.naturalWidth} height={image.naturalHeight} />
        </Layer>
        {showMask && (
          <Layer listening={false}>
            {detections.map((d, i) =>
              d.mask_url ? <MaskImage key={`mask-${i}`} url={d.mask_url} /> : null,
            )}
          </Layer>
        )}
        {showBbox && (
          <Layer listening={false}>
            {detections.map((d, i) => {
              const [x1, y1, x2, y2] = d.bbox;
              const color = COLORS[i % COLORS.length];
              const isSelected = selectedIndex === i;
              return (
                <Fragment key={`det-${i}`}>
                  <Rect
                    x={x1}
                    y={y1}
                    width={x2 - x1}
                    height={y2 - y1}
                    stroke={color}
                    strokeWidth={(isSelected ? 4 : 2) / scale}
                  />
                  <Text
                    x={x1}
                    y={Math.max(0, y1 - 18 / scale)}
                    text={`${d.class_kr} ${(d.score * 100).toFixed(0)}%`}
                    fontSize={14 / scale}
                    fill={color}
                    fontStyle="bold"
                  />
                </Fragment>
              );
            })}
          </Layer>
        )}
      </Stage>
    </div>
  );
}
