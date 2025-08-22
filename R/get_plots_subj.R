library(dplyr)
library(tidyr)
library(ggplot2)
library(scales)
library(showtext)
library(sysfonts)

font_add(
  family  = "CMU Serif",
  regular = "/Users/rushil/Downloads/cmu-serif/cmunrm.ttf",
  bold    = "/Users/rushil/Downloads/cmu-serif/cmunbx.ttf"
)
showtext_auto()
showtext_opts(dpi = 300)  # good for consistent sizing in saved outputs

# --- Load updated QC CSVs ---
overall   <- read.csv("/Users/rushil/ichseg/local_results/subj_lvl/Rushil_QC_subj_results.csv")
artifact  <- read.csv("/Users/rushil/ichseg/local_results/subj_lvl/Rushil_QC_artifact_subj_results.csv")
craniot   <- read.csv("/Users/rushil/ichseg/local_results/subj_lvl/Rushil_QC_craniotomy_subj_results.csv")
cta       <- read.csv("/Users/rushil/ichseg/local_results/subj_lvl/Rushil_QC_cta_subj_results.csv")

# Greyscale fills, consistent with prior figures
fill_methods <- c(
  "v1"         = "#444448",
  "robust"     = "#666666",
  "synthstrip" = "#888888",
  "hdctbet"    = "#AAAAAA",
  "ctbet"      = "#CCCCCC",
  "brainchop"  = "#EEEEEE",
  "dockerctbet" = "#DDDDDD"
)

fill_labels <- c(
  "v1"         = "CTBET",
  "robust"     = "Robust-CTBET",
  "synthstrip" = "SynthStrip",
  "hdctbet"    = "HD-CTBET",
  "ctbet"      = "CT_BET",
  "brainchop"  = "Brainchop",
  "dockerctbet"= "CTbet-Docker"
)

# Helper theme (Computer Modern look if showtext is enabled)
theme_cm <- theme_minimal(base_size = 14, base_family = "CMU Serif") +
  theme(
    panel.grid.major.x = element_line(color = "grey80"),
    panel.grid.major.y = element_line(color = "grey80"),
    panel.grid.minor   = element_blank(),
    axis.line.x        = element_line(color = "black"),
    axis.ticks.x       = element_line(color = "black"),
    axis.line.y        = element_blank(),
    axis.title.x       = element_text(face = "bold", margin = margin(t = 10)),
    axis.title.y       = element_text(face = "bold", margin = margin(r = 10)),
    axis.text.x        = element_text(size = 12, color = "black"),
    axis.text.y        = element_text(size = 12, color = "black"),
    legend.position    = "bottom",
    legend.text        = element_text(size = 12)
  )

fill_methods_std <- setNames(unname(fill_methods[names(fill_labels)]), fill_labels)

# ========== FIGURE A: Overall (single-pass) failure rates by method ==========
overall2 <- overall %>%
  arrange(Total_Failure_Rate) %>%
  mutate(
    Method = factor(Method, levels = Method),
    Method = recode(Method, !!!fill_labels)  # now Method has new names
  )

pA <- ggplot(overall2, aes(x = Method, y = Total_Failure_Rate, fill = Method)) +
  geom_hline(yintercept = 0, color = "black") +
  geom_col(width = 0.6, color = "black", size = 0.8, alpha = 0.9) +
  geom_text(aes(label = sprintf("%.2f", Total_Failure_Rate)),
            vjust = -0.3, fontface = "bold", size = 5, family = "CMU Serif") +
  scale_fill_manual(values = fill_methods_std, guide = "none") +   # <-- use std
  scale_y_continuous(expand = expansion(add = c(0, 5)), breaks = pretty_breaks(10)) +
  labs(x = "Method", y = "Exclusion Rate (%)") +
  theme_cm +
  theme(axis.text.x = element_text(face = "bold"))

ggsave("figureA_overall_exclusion_rates.pdf", pA, width = 12, height = 6, units = "in", dpi = 300, device = cairo_pdf)

# ========== FIGURE B: Task-specific exclusion rates by method ==========
method_order <- overall %>% arrange(Total_Failure_Rate) %>% pull(Method)

tasks_long <- overall %>%
  transmute(
    Method,
    Registration = Registration_Rate,
    Volumetrics  = Volumetrics_Rate,
    `Deep-Learning` = DL_Rate
  ) %>%
  pivot_longer(-Method, names_to = "Task", values_to = "Rate") %>%
  mutate(
    Method = factor(Method, levels = method_order),
    Method = recode(Method, !!!fill_labels),   # <-- display names on x-axis
    Task   = factor(Task, levels = c("Registration","Volumetrics","Deep-Learning"))
  )

pB <- ggplot(tasks_long, aes(x = Method, y = Rate, fill = Task)) +
  geom_hline(yintercept = 0, color = "black") +
  geom_col(position = position_dodge(width = 0.7), width = 0.6,
           color = "black", size = 0.8, alpha = 0.9) +
  geom_text(aes(label = sprintf("%.2f", Rate)),
            position = position_dodge(width = 0.7), vjust = -0.35,
            fontface = "bold", size = 3, family = "CMU Serif") +
  scale_fill_manual(values = fill_tasks, name = NULL) +   # <-- no fill_labels here
  scale_y_continuous(expand = expansion(add = c(0, 5)), breaks = pretty_breaks(8)) +
  labs(x = "Method", y = "Exclusion Rate (%)") +
  theme_cm +
  theme(axis.text.x = element_text(face = "bold"))

ggsave("figureB_task_exclusion_rates.pdf", pB, width = 12, height = 6, units = "in", dpi = 300, device = cairo_pdf)

# ========== FIGURE C: Multiple-failures rate by method (optional) ==========
overall_mult <- overall %>%
  arrange(Multiple_Failures_Rate) %>%
  mutate(
    Method = factor(Method, levels = Method),
    Method = recode(Method, !!!fill_labels)   # <-- recode so x-axis matches
  )

pC <- ggplot(overall_mult, aes(x = Method, y = Multiple_Failures_Rate, fill = Method)) +
  geom_hline(yintercept = 0, color = "black") +
  geom_col(width = 0.6, color = "black", size = 0.8, alpha = 0.9) +
  geom_text(aes(label = sprintf("%.2f", Multiple_Failures_Rate)),
            vjust = -0.3, fontface = "bold", size = 5, family = "CMU Serif") +
  scale_fill_manual(values = fill_methods_std, guide = "none") +   # <-- use std
  scale_y_continuous(expand = expansion(add = c(0, 5)), breaks = pretty_breaks(8)) +
  labs(x = "Method", y = "Multiple-Task Exclusion Rate (%)") +
  theme_cm + 
  theme(axis.text.x = element_text(face = "bold"))

ggsave("figureC_multiple_failures_rates.pdf", pC, width = 12, height = 6, units = "in", dpi = 300, device = cairo_pdf)


# ========== FIGURE D: Subgroup comparisons (Artifact, Craniotomy, CTA) ==========
# Helper to pivot and plot subgroup (expects only methods you want to compare, e.g., v1 vs robust)
plot_subgroup <- function(df, title_file) {
  df <- df %>%
    arrange(Total_Failure_Rate) %>%
    mutate(
      Method = factor(Method, levels = Method),
      Method = recode(Method, !!!fill_labels)
    )
  
  long <- df %>%
    transmute(
      Method,
      Registration = Registration_Rate,
      Volumetrics  = Volumetrics_Rate,
      `Deep-Learning` = DL_Rate
    ) %>%
    tidyr::pivot_longer(-Method, names_to = "Task", values_to = "Rate") %>%
    dplyr::mutate(Task = factor(Task, levels = c("Registration","Volumetrics","Deep-Learning")))
  
  gg <- ggplot(long, aes(x = Method, y = Rate, fill = Task)) +
    geom_hline(yintercept = 0, color = "black") +
    geom_col(position = position_dodge(width = 0.7), width = 0.6,
             color = "black", size = 0.8, alpha = 0.9) +
    geom_text(aes(label = sprintf("%.2f", Rate)),
              position = position_dodge(width = 0.7), vjust = -0.35,
              fontface = "bold", size = 3, family = "CMU Serif") +
    scale_fill_manual(values = fill_tasks, name = NULL) +  # <-- no fill_labels here
    scale_y_continuous(expand = expansion(add = c(0, 5)), breaks = pretty_breaks(8)) +
    labs(x = NULL, y = "Exclusion Rate (%)") +
    theme_cm +
    theme(axis.text.x = element_text(face = "bold"))
  
  ggsave(title_file, gg, width = 12, height = 6, units = "in", dpi = 300, device = cairo_pdf)
}

plot_subgroup(artifact,  "figureD_subgroup_artifact.pdf")
plot_subgroup(craniot,   "figureE_subgroup_craniotomy.pdf")
plot_subgroup(cta,       "figureF_subgroup_cta.pdf")