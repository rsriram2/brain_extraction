library(dplyr)
library(tidyr)
library(readr)
library(stringr)
library(purrr)

remove_brackets    <- function(x) gsub("\\[|\\]", "", x)
remove_back_slash  <- function(x) gsub("\\\\", "", x)
remove_forward_slash <- function(x) gsub("/", "", x)

result <- read_csv('/Users/rushil/ichseg/header_data_cleaned.csv')

out <- result %>%
  filter(! id_patient_short %in% c("6046","6084","6096","6246","6315","6342","6499")) %>%
  mutate(
    Manufacturer = tolower(Manufacturer),
    Manufacturer = sub(";.*", "", Manufacturer),
    Manufacturer = remove_brackets(Manufacturer)
  ) %>%
  mutate(
    KVP = sub(";.*", "", KVP),
    KVP = remove_brackets(KVP),
    KVP = as.numeric(KVP)
  ) %>%
  mutate(
    GantryDetectorTilt = sub(";.*", "", GantryDetectorTilt),
    GantryDetectorTilt = remove_brackets(GantryDetectorTilt),
    GantryDetectorTilt = as.numeric(GantryDetectorTilt),
    has_gantry_tilt    = GantryDetectorTilt != 0
  ) %>%
  mutate(
    PixelSpacing_raw = sub(";.*", "", PixelSpacing),
    PixelSpacing_raw = remove_brackets(PixelSpacing_raw),
    PixelSpacing_raw = remove_back_slash(PixelSpacing_raw),
    PixelSpacing_raw = remove_forward_slash(PixelSpacing_raw)
  ) %>%
  mutate(
    nums = str_extract_all(
      PixelSpacing_raw,
      "\\d*\\.?\\d+(?:[eE][+-]?\\d+)?"
    ),
    PixelSpacing1 = map_dbl(nums, ~ if (length(.x)>=1) as.numeric(.x[[1]]) else NA_real_),
    PixelSpacing2 = map_dbl(nums, ~ if (length(.x)>=2) as.numeric(.x[[2]]) else NA_real_)
  ) %>%
  select(-nums, -PixelSpacing_raw)

# summaries
out %>% count(Manufacturer)
out %>% count(KVP)
out %>% count(has_gantry_tilt)
out %>% summarise(
  mean_ps1 = mean(PixelSpacing1),
  mean_ps2 = mean(PixelSpacing2)
)


out %>%
  filter(is.na(Manufacturer) | grepl("^\\d", Manufacturer)) %>% 
  pull(file_header_wide)
lapply(out %>%
         filter(is.na(Manufacturer) | grepl("^\\d", Manufacturer)) %>% 
         pull(file_header_wide), function(x) unique(readRDS(x)$Manufacturer))
out %>%
  filter(is.na(KVP))
lapply(out %>%
         filter(is.na(KVP)) %>% 
         pull(file_header_wide), function(x) unique(readRDS(x)$KVP))
out %>%
  filter(is.na(GantryDetectorTilt))
lapply(out %>%
         filter(is.na(GantryDetectorTilt)) %>% 
         pull(file_header_wide), function(x) unique(readRDS(x)$GantryDetectorTilt))
out %>%
  filter(is.na(PixelSpacing))