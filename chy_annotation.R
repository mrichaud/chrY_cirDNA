# Load required libraries
library(tidyverse)
library(GenomicRanges)

# -----------------------------------------------------------------------------
# 1. Recreate the Subregion Data
# -----------------------------------------------------------------------------
subregions <- tibble::tribble(
  ~Subregion_type,   ~Subregion_abbreviation, ~Colour_code, ~chromosome, ~start,    ~end,
  "Pseudo-autosomal", "PAR1",                 "#97CB99",    "chrY",      10000,    2781479,
  "X-degenerate",     "XDR1",                 "#FFEF57",    "chrY",      2781480,  3049682,
  "X-transposed",     "XTR1",                 "#EEA9BA",    "chrY",      3049683,  6234809,
  "Ampliconic",       "AMPL1",                "#88C0EA",    "chrY",      6234810,  6532527,
  "X-transposed",     "XTR2",                 "#EEA9BA",    "chrY",      6532528,  6748297,
  "X-degenerate",     "XDR2",                 "#FFEF57",    "chrY",      6748298,  7586094,
  "Ampliconic",       "AMPL2",                "#88C0EA",    "chrY",      7586095,  10130374,
  "Others",           "other1",               "#D9D8DB",    "chrY",      10130375, 10197195,
  "Heterochromatic",  "HET1_centro1",         "#777777",    "chrY",      10197196, 10203018,
  "Heterochromatic",  "HET1_centroDYZ3",      "#7E2341",    "chrY",      10203019, 10626584,
  "Heterochromatic",  "HET1_centro2",         "#777777",    "chrY",      10626585, 11749309,
  "X-degenerate",     "XDR3",                 "#FFEF57",    "chrY",      11749310, 13984445,
  "Ampliconic",       "AMPL3",                "#88C0EA",    "chrY",      13984446, 14058287,
  "X-degenerate",     "XDR4",                 "#FFEF57",    "chrY",      14058288, 15874849,
  "Ampliconic",       "AMPL4",                "#88C0EA",    "chrY",      15874850, 15904951,
  "X-degenerate",     "XDR5",                 "#FFEF57",    "chrY",      15904952, 16159547,
  "Ampliconic",       "AMPL5",                "#88C0EA",    "chrY",      16159548, 16425803,
  "X-degenerate",     "XDR6",                 "#FFEF57",    "chrY",      16425804, 17455802,
  "Ampliconic",       "AMPL6",                "#88C0EA",    "chrY",      17455803, 18870186,
  "X-degenerate",     "XDR7",                 "#FFEF57",    "chrY",      18870187, 20054847,
  "Heterochromatic",  "HET2_DYZ19",           "#777777",    "chrY",      20054848, 20351037,
  "X-degenerate",     "XDR8",                 "#FFEF57",    "chrY",      20351038, 21335746,
  "Ampliconic",       "AMPL7",                "#88C0EA",    "chrY",      21335747, 26644162,
  "Others",           "other2_DYZ18",         "#D9D8DB",    "chrY",      26644163, 27078487,
  "Heterochromatic",  "HET3_Yq",              "#777777",    "chrY",      27078488, 56887901,
  "Pseudo-autosomal", "PAR2",                 "#97CB99",    "chrY",      56887902, 57217415
)

# -----------------------------------------------------------------------------
# 2. Generate 10 kb Bins Across Chromosome Y
# -----------------------------------------------------------------------------
bin_size <- 10000  # 10 kb
chrY_length <- max(subregions$end)

bins <- tibble(
  seqnames = "chrY",
  start = seq(1, chrY_length, by = bin_size),
  end = pmin(seq(bin_size, chrY_length + bin_size - 1, by = bin_size), chrY_length)
)

# -----------------------------------------------------------------------------
# 3. Map Bins to Subregions Using GenomicRanges
# -----------------------------------------------------------------------------
subregions_gr <- GRanges(
  seqnames = subregions$chromosome,
  ranges = IRanges(start = subregions$start, end = subregions$end),
  Subregion_type = subregions$Subregion_type
)

bin_midpoints <- GRanges(
  seqnames = bins$seqnames,
  ranges = IRanges(start = (bins$start + bins$end) / 2, width = 1)
)

hits <- findOverlaps(bin_midpoints, subregions_gr)

# Annotate bins with matched type
binned_ideogram <- bins
binned_ideogram$Subregion_type <- NA_character_
binned_ideogram$Subregion_type[queryHits(hits)] <- subregions_gr$Subregion_type[subjectHits(hits)]

# -----------------------------------------------------------------------------
# 4. Create Color Palette Mapping
# -----------------------------------------------------------------------------
type_colors <- subregions %>%
  select(Subregion_type, Colour_code) %>%
  distinct() %>%
  deframe()

# -----------------------------------------------------------------------------
# 5. Plot Ideogram with ggplot2 (Fixed Warning & NA Color)
# -----------------------------------------------------------------------------
chr_start_mb <- min(binned_ideogram$start) / 1e6
chr_end_mb   <- max(binned_ideogram$end) / 1e6

ggplot(binned_ideogram) +
  # Draw 10 kb bins
  geom_rect(
    aes(xmin = start / 1e6, xmax = end / 1e6, ymin = 0.2, ymax = 0.8, fill = Subregion_type),
    color = NA
  ) +
  # Use annotate() instead of geom_rect for single outer border (fixes warning)
  annotate(
    "rect",
    xmin = chr_start_mb, xmax = chr_end_mb, ymin = 0.2, ymax = 0.8,
    fill = NA, color = "black", linewidth = 0.5
  ) +
  # Explicit na.value makes NA regions distinctly light grey/white vs dark grey Heterochromatic
  scale_fill_manual(
    values = type_colors, 
    na.value = "#F0F0F0", 
    name = "Subregion Type"
  ) +
  scale_x_continuous(
    name = "Position on Chromosome Y (Mb)",
    breaks = seq(0, 60, by = 10),
    expand = c(0.01, 0.01)
  ) +
  scale_y_continuous(limits = c(0, 1)) +
  labs(
    title = "Chromosome Y Ideogram (10 kb Bins)",
    subtitle = "Subregion Annotations Map"
  ) +
  theme_minimal() +
  theme(
    axis.text.y = element_blank(),
    axis.ticks.y = element_blank(),
    axis.title.y = element_blank(),
    panel.grid.major.y = element_blank(),
    panel.grid.minor.y = element_blank(),
    plot.title = element_text(face = "bold", size = 14),
    legend.position = "bottom"
  )