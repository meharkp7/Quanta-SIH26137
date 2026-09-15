import * as React from "react";
import * as SeparatorPrimitive from "@radix-ui/react-separator";
import { cn } from "@/lib/utils";

export const Separator = React.forwardRef<React.ElementRef<typeof SeparatorPrimitive.Root>, React.ComponentPropsWithoutRef<typeof SeparatorPrimitive.Root>>(({ className, orientation = "horizontal", ...props }, ref) => <SeparatorPrimitive.Root ref={ref} orientation={orientation} className={cn("ui-separator", orientation === "vertical" && "ui-separator-vertical", className)} {...props} />);
Separator.displayName = SeparatorPrimitive.Root.displayName;
