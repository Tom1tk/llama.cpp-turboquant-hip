/*
 * triattention-backend.c — resolve HIP acceleration at init time
 */
#include "triattention-backend.h"
#include <string.h>

struct tria_backend g_tria_backend;

#if defined(GGML_USE_HIP) && defined(GGML_USE_CUDA)
/* Static HIP linking — GGML_USE_HIP is always set for HIP builds,
   GGML_USE_CUDA is the compat macro only set when GGML_BACKEND_DL=OFF.
   Both together = HIP static build = safe to resolve tria_hip_* symbols. */
#include "triattention-hip.h"

int tria_backend_init(void) {
    g_tria_backend.stats_upload    = tria_hip_stats_upload;
    g_tria_backend.stats_free      = tria_hip_stats_free;
    g_tria_backend.score_q8_0      = tria_hip_score_q8_0;
    g_tria_backend.scores_download = tria_hip_scores_download;
    g_tria_backend.compact_rows    = tria_hip_compact_rows;
    return 1;
}

#else
/* Dynamic loading or no HIP — CPU fallback (no GPU scoring) */

int tria_backend_init(void) {
    memset(&g_tria_backend, 0, sizeof(g_tria_backend));
    return 0;
}

#endif
