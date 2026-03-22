import type { SelectHTMLAttributes } from "react";

import { cn } from "@/lib/utils";

export function Select({ className, ...props }: SelectHTMLAttributes<HTMLSelectElement>) {
  return (
    <select
      className={cn(
        "w-full rounded-md border border-border bg-card px-3 py-2 text-sm outline-none ring-primary/25 transition focus-visible:ring",
        className,
      )}
      {...props}
    />
  );
}
