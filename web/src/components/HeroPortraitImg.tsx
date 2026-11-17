import { useEffect, useState } from "react";

interface HeroPortraitImgProps {
  src: string | null | undefined;
  alt: string;
  fallbackText: string;
  width: number;
  height: number;
  className?: string;
  decorative?: boolean;
  priority?: boolean;
}

export function HeroPortraitImg({
  src,
  alt,
  fallbackText,
  width,
  height,
  className = "",
  decorative = false,
  priority = false,
}: HeroPortraitImgProps) {
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    setFailed(false);
  }, [src]);

  if (!src || failed) {
    return (
      <span
        className={`asset-fallback ${className}`.trim()}
        aria-hidden={decorative ? true : undefined}
        title={decorative ? undefined : alt}
        style={{ width: `${width}px`, height: `${height}px` }}
      >
        {fallbackText}
      </span>
    );
  }

  return (
    <img
      src={src}
      alt={decorative ? "" : alt}
      width={width}
      height={height}
      loading={priority ? "eager" : "lazy"}
      fetchPriority={priority ? "high" : undefined}
      decoding={priority ? "sync" : "async"}
      className={className}
      onError={() => setFailed(true)}
    />
  );
}
