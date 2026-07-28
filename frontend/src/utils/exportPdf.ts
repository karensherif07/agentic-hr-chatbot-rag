import { jsPDF } from "jspdf";
import type { ChatMessage } from "../api/types";

/**
 * Exports the current conversation as a downloadable PDF. Pure client-side —
 * no backend involvement, so it works instantly on whatever's currently on
 * screen. Arabic/Franco text renders left-to-right in the PDF (jsPDF's
 * built-in fonts don't support Arabic script shaping), so Arabic messages
 * are labeled clearly but may not render Arabic glyphs correctly — this is
 * a known limitation of client-side PDF generation without a bundled
 * Arabic-capable font. English and Franco-Arabic (Latin script) render fine.
 */
export function exportChatAsPdf(history: ChatMessage[], employeeName?: string) {
  const doc = new jsPDF({ unit: "pt", format: "a4" });
  const marginX = 48;
  const pageWidth = doc.internal.pageSize.getWidth();
  const pageHeight = doc.internal.pageSize.getHeight();
  const maxWidth = pageWidth - marginX * 2;
  let y = 56;

  doc.setFont("helvetica", "bold");
  doc.setFontSize(16);
  doc.text("HR Assistant — Conversation Transcript", marginX, y);
  y += 22;

  doc.setFont("helvetica", "normal");
  doc.setFontSize(10);
  doc.setTextColor(110, 110, 110);
  const meta = [
    employeeName ? `Employee: ${employeeName}` : null,
    `Exported: ${new Date().toLocaleString()}`,
  ].filter(Boolean).join("   |   ");
  doc.text(meta, marginX, y);
  y += 20;

  doc.setDrawColor(200, 200, 200);
  doc.line(marginX, y, pageWidth - marginX, y);
  y += 20;

  function ensureSpace(lines: number, lineHeight: number) {
    if (y + lines * lineHeight > pageHeight - 48) {
      doc.addPage();
      y = 56;
    }
  }

  for (const msg of history) {
    const label = msg.role === "user" ? "You" : "HR Assistant";
    const isNonLatin = msg.is_arabic && !msg.is_franco;

    doc.setFont("helvetica", "bold");
    doc.setFontSize(10.5);
    doc.setTextColor(msg.role === "user" ? 40 : 20, msg.role === "user" ? 60 : 20, msg.role === "user" ? 150 : 20);
    ensureSpace(1, 14);
    doc.text(label + (isNonLatin ? "  (Arabic — see note below)" : ""), marginX, y);
    y += 16;

    doc.setFont("helvetica", "normal");
    doc.setFontSize(10);
    doc.setTextColor(30, 30, 30);
    const content = isNonLatin
      ? "[Arabic-script message — open the app to view; PDF export does not render Arabic glyphs]"
      : msg.content;
    const wrapped = doc.splitTextToSize(content, maxWidth);
    ensureSpace(wrapped.length, 13);
    doc.text(wrapped, marginX, y);
    y += wrapped.length * 13 + 14;
  }

  const filenameDate = new Date().toISOString().slice(0, 10);
  doc.save(`hr-assistant-chat-${filenameDate}.pdf`);
}