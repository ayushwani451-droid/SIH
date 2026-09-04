/* Shared compliance-report rendering, extracted from new-compliance-scan.html (Phase 7 /
   Step 3) so it's used identically wherever a compliance report is shown - the New Scan
   page (with a live image + bounding boxes) and the historical Reports page (no stored
   image - see compliance-report.html's note on why). One implementation, not two that can
   drift out of sync.

   applyComplianceColoring/wireChecklistInteractivity are optional: pass previewWrap (and
   previewCard, for the click-to-scroll behavior) only on pages that actually have a live
   .ocr-bbox overlay to color/link against. Omit them and renderComplianceReport() just
   renders the banner/violations/checklist with no box-linking - exactly what a page with
   no image needs. */
(function (global) {
  "use strict";

  function verdictMeta(verdict) {
    if (verdict === 'Pass') {
      return { icon: 'check_circle', color: '#006a6a', textClass: 'text-secondary', pillClass: 'bg-secondary/10 text-secondary', label: 'Compliant' };
    }
    if (verdict === 'Fail') {
      return { icon: 'cancel', color: '#ba1a1a', textClass: 'text-error', pillClass: 'bg-error/10 text-error', label: 'Non-Compliant' };
    }
    if (verdict === 'Needs Review') {
      return { icon: 'warning', color: '#d39d00', textClass: 'text-on-tertiary-container', pillClass: 'bg-tertiary-fixed-dim/25 text-on-tertiary-container', label: 'Needs Review' };
    }
    return { icon: 'remove_circle', color: '#737780', textClass: 'text-on-surface-variant', pillClass: 'bg-outline-variant/30 text-on-surface-variant', label: 'Not Applicable' };
  }

  function overallMeta(overallStatus) {
    if (overallStatus === 'Compliant') {
      return { label: 'COMPLIANT', icon: 'verified', textClass: 'text-secondary', bannerClass: 'bg-secondary/10 border-secondary' };
    }
    if (overallStatus === 'Non-Compliant') {
      return { label: 'NON-COMPLIANT', icon: 'gpp_bad', textClass: 'text-error', bannerClass: 'bg-error/10 border-error' };
    }
    if (overallStatus === 'Partially Compliant') {
      return { label: 'NEEDS REVIEW', icon: 'warning', textClass: 'text-on-tertiary-container', bannerClass: 'bg-tertiary-fixed-dim/15 border-tertiary-fixed-dim' };
    }
    return { label: 'UNKNOWN', icon: 'help', textClass: 'text-on-surface-variant', bannerClass: 'bg-surface-container border-outline-variant' };
  }

  // Draws one uniform-color box per raw OCR detection, absolutely positioned over
  // previewImg inside previewWrap. Each box is tagged with its original (unscaled) bbox
  // so applyComplianceColoring() can find and recolor it later.
  function drawBoundingBoxes(detections, previewImg, previewWrap) {
    if (!detections || !detections.length) return;
    var naturalW = previewImg.naturalWidth;
    var naturalH = previewImg.naturalHeight;
    var renderedW = previewImg.clientWidth;
    var renderedH = previewImg.clientHeight;
    if (!naturalW || !naturalH) return;
    var scaleX = renderedW / naturalW;
    var scaleY = renderedH / naturalH;

    detections.forEach(function (det) {
      var box = document.createElement('div');
      box.className = 'ocr-bbox';
      var x1 = det.bbox[0] * scaleX, y1 = det.bbox[1] * scaleY;
      var x2 = det.bbox[2] * scaleX, y2 = det.bbox[3] * scaleY;
      box.setAttribute('style',
        'position:absolute;left:' + x1 + 'px;top:' + y1 + 'px;' +
        'width:' + (x2 - x1) + 'px;height:' + (y2 - y1) + 'px;' +
        'border:2px solid #00274a;background:rgba(0,39,74,0.08);pointer-events:none;' +
        'box-sizing:border-box;transition:border-color .2s ease,background-color .2s ease;');
      box.title = det.text + ' (' + Math.round(det.confidence * 100) + '%)';
      box.dataset.bbox = det.bbox.join(',');
      previewWrap.appendChild(box);
    });
  }

  // Re-colors the SAME .ocr-bbox divs drawBoundingBoxes() already created (matched by
  // their stored original bbox), rather than rebuilding the overlay - boxes not tied
  // to any Rule 6 field keep their original uniform navy color.
  function applyComplianceColoring(report, previewWrap) {
    var boxes = previewWrap.querySelectorAll('.ocr-bbox');
    var boxByBbox = {};
    boxes.forEach(function (box) {
      box.style.border = '2px solid #00274a';
      box.style.background = 'rgba(0,39,74,0.08)';
      box.removeAttribute('data-field');
      boxByBbox[box.dataset.bbox] = box;
    });

    (report.fields || []).forEach(function (field) {
      var meta = verdictMeta(field.verdict);
      (field.matched_detections || []).forEach(function (det) {
        var box = boxByBbox[(det.bbox || []).join(',')];
        if (!box) return;
        box.style.borderColor = meta.color;
        box.style.borderWidth = '3px';
        box.style.background = meta.color + '22';
        box.dataset.field = field.field;
        box.title = det.text + ' — ' + field.field_label + ': ' + meta.label;
      });
    });
  }

  // Fallback for connector lines (see the .ocr-bbox-pulse CSS rule wherever it's defined):
  // matching colors plus hover/click highlighting links a checklist row to its box(es)
  // instead of drawing lines that would often point off-screen in a scrolling layout.
  function wireChecklistInteractivity(containerEl, previewWrap, previewCard) {
    var rows = containerEl.querySelectorAll('[data-checklist-row]');
    rows.forEach(function (row) {
      var fieldKey = row.getAttribute('data-field');
      var boxes = previewWrap.querySelectorAll('.ocr-bbox[data-field="' + fieldKey + '"]');
      if (!boxes.length) return;

      function pulse(duration) {
        var color = global.getComputedStyle(row).borderLeftColor;
        boxes.forEach(function (b) {
          b.style.color = color;
          b.classList.add('ocr-bbox-pulse');
        });
        if (duration) {
          setTimeout(function () { boxes.forEach(function (b) { b.classList.remove('ocr-bbox-pulse'); }); }, duration);
        }
      }

      row.addEventListener('mouseenter', function () { row.classList.add('compliance-row-active'); pulse(); });
      row.addEventListener('mouseleave', function () {
        row.classList.remove('compliance-row-active');
        boxes.forEach(function (b) { b.classList.remove('ocr-bbox-pulse'); });
      });
      row.addEventListener('click', function () {
        if (previewCard) previewCard.scrollIntoView({ behavior: 'smooth', block: 'center' });
        pulse(2000);
      });
    });
  }

  // Renders the verdict banner + violations list + Rule 6 checklist into containerEl.
  // Pass { previewWrap, previewCard } to also color-code and link an existing .ocr-bbox
  // overlay; omit it entirely on a page with no image (matched_detections just won't be
  // interactive, which is the correct, honest behavior when there's nothing to point at).
  function renderComplianceReport(report, containerEl, opts) {
    opts = opts || {};
    var escapeHtml = (global.LL && global.LL.escapeHtml) || function (s) { return String(s == null ? '' : s); };

    var overall = overallMeta(report.overall_status);
    var violationCount = (report.violations || []).length;
    var summaryLine = report.overall_status === 'Compliant'
      ? 'Meets Legal Metrology (Packaged Commodities) Rules, 2011.'
      : report.overall_status === 'Unknown'
        ? 'Compliance could not be evaluated for this image.'
        : violationCount + ' issue' + (violationCount === 1 ? '' : 's') + ' found requiring attention.';

    var bannerHtml =
      '<div class="rounded-xl border-2 p-6 flex items-center gap-4 mb-6 ' + overall.bannerClass + '">' +
      '<span class="material-symbols-outlined text-5xl ' + overall.textClass + '" style="font-variation-settings:&quot;FILL&quot; 1">' + overall.icon + '</span>' +
      '<div><div class="text-headline-lg font-headline-lg font-black tracking-wide ' + overall.textClass + '">' + overall.label + '</div>' +
      '<p class="text-body-lg font-body-lg text-on-surface-variant mt-1">' + escapeHtml(summaryLine) + '</p></div></div>';

    var violationsHtml = '';
    if (violationCount > 0) {
      violationsHtml = '<div class="mb-6"><h4 class="text-body-lg font-body-lg font-bold text-on-surface mb-3">Violations &amp; Items Needing Review</h4>' +
        '<div class="flex flex-col gap-2">' +
        report.violations.map(function (v) {
          var isHardFail = v.severity === 'violation';
          return '<div class="flex items-start gap-3 p-3 rounded-lg ' + (isHardFail ? 'bg-error/5' : 'bg-tertiary-fixed-dim/10') + '">' +
            '<span class="material-symbols-outlined ' + (isHardFail ? 'text-error' : 'text-on-tertiary-container') + '">' + (isHardFail ? 'error' : 'warning') + '</span>' +
            '<div><p class="font-bold text-on-surface">' + escapeHtml(v.field_label) + '</p>' +
            '<p class="text-body-md font-body-md text-on-surface-variant">' + escapeHtml(v.reason) + '</p></div></div>';
        }).join('') + '</div></div>';
    } else if (report.fields && report.fields.length) {
      violationsHtml = '<div class="mb-6 flex items-center gap-3 text-secondary">' +
        '<span class="material-symbols-outlined">task_alt</span><p class="font-bold">No violations found.</p></div>';
    }

    var checklistHtml = '';
    if (report.fields && report.fields.length) {
      checklistHtml = '<div><h4 class="text-body-lg font-body-lg font-bold text-on-surface mb-3">Rule 6 Declaration Checklist</h4>' +
        '<div class="flex flex-col">' +
        report.fields.map(function (field, idx) {
          var meta = verdictMeta(field.verdict);
          var hasBoxes = opts.previewWrap && field.matched_detections && field.matched_detections.length > 0;
          return '<div class="flex items-start gap-3 py-3 px-3 -mx-3 rounded-lg border-b border-outline-variant/50 last:border-0 transition-colors' + (hasBoxes ? ' cursor-pointer hover:bg-surface-container-low' : '') + '" ' +
            'data-checklist-row="' + idx + '" data-field="' + escapeHtml(field.field) + '" style="border-left:4px solid ' + meta.color + '">' +
            '<span class="material-symbols-outlined ' + meta.textClass + '" style="font-variation-settings:&quot;FILL&quot; 1">' + meta.icon + '</span>' +
            '<div class="flex-1 min-w-0">' +
            '<div class="flex items-center justify-between gap-2 flex-wrap">' +
            '<span class="font-bold text-on-surface">' + escapeHtml(field.field_label) +
            (field.conditional_field ? ' <span class="text-label-md font-label-md text-on-surface-variant font-normal">(conditional)</span>' : '') + '</span>' +
            '<span class="text-label-md font-label-md font-bold px-2 py-0.5 rounded-full ' + meta.pillClass + '">' + meta.label + '</span>' +
            '</div>' +
            '<p class="text-body-md font-body-md text-on-surface-variant mt-0.5">' + (field.extracted_value ? escapeHtml(field.extracted_value) : '<em>Not detected</em>') + '</p>' +
            (field.reason ? '<p class="text-label-md font-label-md text-on-surface-variant italic mt-1">' + escapeHtml(field.reason) + '</p>' : '') +
            '</div></div>';
        }).join('') + '</div></div>';
    } else if (report.notes && report.notes.length) {
      checklistHtml = '<div class="flex items-start gap-3 text-on-surface-variant"><span class="material-symbols-outlined">info</span>' +
        '<p>' + escapeHtml(report.notes.join(' ')) + '</p></div>';
    }

    containerEl.innerHTML = bannerHtml + violationsHtml + checklistHtml;

    if (opts.previewWrap) {
      applyComplianceColoring(report, opts.previewWrap);
      wireChecklistInteractivity(containerEl, opts.previewWrap, opts.previewCard);
    }
  }

  global.LLComplianceUI = {
    verdictMeta: verdictMeta,
    overallMeta: overallMeta,
    drawBoundingBoxes: drawBoundingBoxes,
    applyComplianceColoring: applyComplianceColoring,
    wireChecklistInteractivity: wireChecklistInteractivity,
    renderComplianceReport: renderComplianceReport,
  };
})(window);
