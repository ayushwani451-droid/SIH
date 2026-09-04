/* Fetch wrapper for GET /api/ocr/compliance/{scan_id}. Kept separate from ocr-client.js
   on purpose: this hits a different endpoint, called only after a scan already exists
   (POST /api/scans is synchronous - there's no polling step to hook into), and
   ocr-client.js's existing OCR flow shouldn't need to know this endpoint exists. */
(function (global) {
  "use strict";

  var DEFAULT_BASE_URL = "http://127.0.0.1:8000";

  function getCompliance(scanId, profile, baseUrl) {
    var url = (baseUrl || DEFAULT_BASE_URL) + "/api/ocr/compliance/" + encodeURIComponent(scanId);
    if (profile) url += "?profile=" + encodeURIComponent(profile);

    return fetch(url, { credentials: "include" })
      .then(function (response) {
        return response.json().catch(function () { return null; }).then(function (data) {
          return { httpOk: response.ok, status: response.status, data: data };
        });
      })
      .catch(function (networkErr) {
        return {
          httpOk: false,
          status: 0,
          data: { detail: "Could not reach the compliance backend at " + url + ". Is it running? (" + networkErr.message + ")" },
        };
      });
  }

  global.LLCompliance = {
    getCompliance: getCompliance,
    DEFAULT_BASE_URL: DEFAULT_BASE_URL,
  };
})(window);
