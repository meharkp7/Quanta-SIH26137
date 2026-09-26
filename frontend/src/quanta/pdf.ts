// Minimal, dependency-free PDF writer for the "Download PDF Action Plan"
// action: one Helvetica text document, paginated. Everything is ASCII after
// escaping, so byte offsets for the xref table equal string indices.

type Line = { text: string; bold?: boolean; size?: number; gap?: number };

const LINES_PER_PAGE = 46;

function esc(value: string): string {
  return value
    .replace(/\\/g, "\\\\")
    .replace(/\(/g, "\\(")
    .replace(/\)/g, "\\)")
    // eslint-disable-next-line no-control-regex
    .replace(/[^\x20-\x7E]/g, "-");
}

function pageStream(title: string, lines: Line[], isFirst: boolean): string {
  const parts: string[] = ["BT", "40 800 Td"];
  if (isFirst) {
    parts.push(`/F2 17 Tf 24 TL (${esc(title)}) T*`);
    parts.push("/F1 9.5 Tf 125 TL (Quanta - adaptive quantum-inspired vehicle routing - SIH26137) T*");
    parts.push("0 -8 TL");
  } else {
    parts.push(`/F2 12 Tf 20 TL (${esc(title)} (continued)) T*`);
    parts.push("/F1 9.5 Tf 0 -6 TL");
  }
  parts.push("/F1 10 Tf 14 TL");
  for (const line of lines) {
    const size = line.size ?? 10;
    if (line.bold) parts.push(`/F2 ${size} Tf`);
    else parts.push(`/F1 ${size} Tf`);
    if (line.gap) parts.push(`${line.gap} TL`);
    parts.push(`(${esc(line.text)}) T*`);
    parts.push("/F1 10 Tf 14 TL");
  }
  parts.push("ET");
  return parts.join("\n");
}

/** Build an application/pdf Blob from a title + line list. */
export function buildPdf(title: string, lines: Line[]): Blob {
  const pages: Line[][] = [];
  for (let i = 0; i < lines.length; i += LINES_PER_PAGE) pages.push(lines.slice(i, i + LINES_PER_PAGE));
  if (!pages.length) pages.push([{ text: "" }]);

  const objects: string[] = [];
  const pageIds: number[] = [];
  const firstObject = 5; // 1 catalog, 2 pages, 3 font regular, 4 font bold …
  const pageCount = pages.length;
  const contentIds = pages.map((_, i) => firstObject + pageCount + i);

  objects.push("<< /Type /Catalog /Pages 2 0 R >>");
  objects.push(`<< /Type /Pages /Kids [${pageIdsPlaceholder(pageCount, firstObject)}] /Count ${pageCount} >>`);
  objects.push("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>");
  objects.push("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>");
  pages.forEach((linesOfPage, i) => {
    objects.push(
      `<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Resources << /Font << /F1 3 0 R /F2 4 0 R >> >> /Contents ${contentIds[i]} 0 R >>`,
    );
  });
  pages.forEach((linesOfPage, i) => {
    const stream = pageStream(title, linesOfPage, i === 0);
    objects.push(`<< /Length ${stream.length} >>\nstream\n${stream}\nendstream`);
  });

  let out = "%PDF-1.4\n";
  const offsets: number[] = [];
  objects.forEach((body, index) => {
    offsets.push(out.length);
    out += `${index + 1} 0 obj\n${body}\nendobj\n`;
  });
  const xrefAt = out.length;
  out += `xref\n0 ${objects.length + 1}\n0000000000 65535 f \n`;
  for (const offset of offsets) out += `${String(offset).padStart(10, "0")} 00000 n \n`;
  out += `trailer\n<< /Size ${objects.length + 1} /Root 1 0 R >>\nstartxref\n${xrefAt}\n%%EOF`;

  const bytes = new Uint8Array(out.length);
  for (let i = 0; i < out.length; i += 1) bytes[i] = out.charCodeAt(i) & 0xff;
  return new Blob([bytes], { type: "application/pdf" });
}

function pageIdsPlaceholder(pageCount: number, firstObject: number): string {
  return Array.from({ length: pageCount }, (_, i) => `${firstObject + i} 0 R`).join(" ");
}

/** Trigger a browser download for a generated blob. */
export function downloadBlob(blob: Blob, fileName: string) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = fileName;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  window.setTimeout(() => URL.revokeObjectURL(url), 4000);
}
