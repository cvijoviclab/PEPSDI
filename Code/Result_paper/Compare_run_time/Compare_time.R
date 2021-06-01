library(tidyverse)
library(stringr)
library(mcmcse)
library(ggthemes)
library(corrplot)
library(data.table)


# General plotting parameters 
my_theme <-  theme_tufte(base_size = 24) + theme(plot.title = element_text(hjust = 0.5, size = 14, face="bold"), 
                                                 plot.subtitle = element_text(hjust = 0.5)) +
  theme(axis.title=element_text(size=24)) + theme(rect = element_rect(fill = "transparent"))
cbPalette <- c(
  "#999999", "#E69F00", "#56B4E9", "#009E73",
  "#F0E442", "#0072B2", "#D55E00", "#CC79A7")
BASE_HEIGHT <- 5
BASE_WIDTH <- 7.0


dir_base <- "../../../Intermediate/Multiple_individuals/Schlogl_benchmark_time/"
file_res <- list.files(dir_base)
res_tibble <- tibble()

# Collect all run-times 
for(i in 1:length(file_res)){
  case_i <- file_res[i]
  
  # Extract relevant data 
  n_ind <- as.integer(str_match(case_i, "N_ind(\\d+)_")[2])
  rho <- as.numeric(str_match(case_i, "rho(\\d|.+)_")[2])
  if(str_detect(case_i, "new")){
    sampler <- "new"
  }else{
    sampler <- "old"
  }
  
  dir_files <- str_c(dir_base, case_i, "/Ram_sampler/")
  file_list <- list.files(dir_files)
  file_list <- file_list[file_list != "Pilot_run_data_alt"]
  file_list <- file_list[file_list != "Pilot_run_data"]
  
  if(length(file_list) == 0){next}
  
  for(j in 1:length(file_list)){
    
    print(case_i)
    print(file_list[j])
    run_time_data <- as.numeric(read_csv(str_c(dir_files, file_list[j], "/Run_time.csv"), 
                                         col_types = cols())[1, 1])
    
    res_save <- tibble(run_time = run_time_data, sampler = sampler, 
                       n_ind = n_ind, rho = rho)
    res_tibble <- res_tibble %>%
      bind_rows(res_save)
  }
  
}


dir_save <- "../../../Results/Paper_figures/Fig_s_time/"
if(!dir.exists(dir_save)) dir.create(dir_save, recursive = T)

res_tibble_plot <- res_tibble %>%
  mutate(sampler = as.factor(sampler), n_ind = as.factor(n_ind), rho = as.factor(rho), 
         run_time = run_time / (1000 * 60)) %>%
  mutate(n_ind_int = as.integer(as.character(n_ind)))


# First plot of run-time 
res_sum <- res_tibble_plot %>%
  group_by(n_ind_int, sampler) %>%
  summarise(mean = mean(run_time), 
            max = max(run_time), 
            min = min(run_time))

p <- ggplot(res_sum, aes(n_ind_int, mean, color = sampler)) + 
  geom_line(size = 1.0) + 
  geom_point(size = 2.0) + 
  geom_linerange(aes(ymin = min, ymax = max, color = sampler), size = 1.5) + 
  scale_color_manual(values = cbPalette[-1]) + 
  labs(y = "Run-time [min]", x = "Number of individuals") + 
  geom_rangeframe(size = 1.0, color = "black") + 
  labs(x = "", y = "") + 
  scale_y_log10() + 
  my_theme + theme(legend.position = "none")
ggsave(str_c(dir_save, "Time_vs_ind.svg"), p, bg = "transparent", width = BASE_WIDTH, height = BASE_HEIGHT)

data_old <- res_sum %>%
  filter(sampler == "old") %>%
  rename("mean_old" = "mean", "max_old" = "max", "min_old" = "min") 
data_new <- res_sum %>%
  filter(sampler == "new") %>%
  rename("mean_new" = "mean", "max_new" = "max", "min_new" = "min") %>%
  filter(n_ind_int < 200) %>% 
  select(-n_ind_int, -sampler)

data_ratio <- data_new %>%
  bind_cols(data_old) %>%
  select(-n_ind_int...5, -n_ind_int...1) %>%
  mutate(ratio_mean = mean_old / mean_new, 
         ratio_max = max_old / max_new, 
         ratio_min = min_old / min_new, 
         n_ind = c(20, 40, 60, 80, 100))

range_data <- tibble(x = c(20, 100), y = c(0, 40))
p <- ggplot(data_ratio, aes(n_ind, ratio_mean)) + 
  geom_line(size = 1.0) + 
  geom_point(size = 2.0) + 
  geom_linerange(aes(ymin = ratio_max, ymax = ratio_min), size = 1.5) +
  scale_color_manual(values = cbPalette[-1]) + 
  scale_fill_manual(values = cbPalette[-1]) + 
  geom_rangeframe(data=range_data, mapping=aes(x, y), size = 1.0) + 
  labs(x = "", y = "") + 
  my_theme + theme(legend.position = "none")
ggsave(str_c(dir_save, "Ratio_run_time.svg"), p, bg = "transparent", width = BASE_WIDTH, height = BASE_HEIGHT)
