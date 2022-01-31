library(tidyverse)
library(ggthemes)
library(latex2exp)
library(data.table)
library(mcmcse)

# General plotting parameters 
my_theme <- theme_tufte(base_size = 24) + theme(plot.title = element_text(hjust = 0.5, size = 14, face="bold"), 
                                                plot.subtitle = element_text(hjust = 0.5)) +
  theme(axis.title=element_text(size=24)) + theme(rect = element_rect(fill = "transparent"))
my_colors <- c(
  "#999999", "#E69F00", "#56B4E9", "#009E73",
  "#F0E442", "#0072B2", "#D55E00", "#CC79A7")
BASE_HEIGHT <- 5
BASE_WIDTH <- 7.0


dir_save <- "../../../Results/Paper_figures/Fig4/"


process_single_individual <- function(model_name)
{
  dir_data <- str_c("../../../Intermediate/Single_individual/", model_name)
  
  data_res <- tibble()
  
  file_list <- list.files(dir_data)
  for(i in 1:length(file_list)){
    
    
    dir_res <- file_list[i]
    path_res <- str_c(dir_data, "/", dir_res, "/")
    
    # Extract start-guess and data-set 
    start_guess <- as.integer(str_match(dir_res, "sg(\\d+)_")[2])
    print(start_guess)
    data_set <- as.integer(str_match(dir_res, "_(\\d+)$")[2])
    
    # Loop through the samplers 
    j <- 1
    for(j in 1:length(list.files(path_res))){
      sampler_res <- list.files(path_res)[j]
      
      if(sampler_res == "AM"){ 
        sampler <- "AM"
        path_files <- str_c(path_res, "AM/Correlated/Am_sampler/")
      }else if(sampler_res == "GenAM"){
        sampler <- "GenAM"
        path_files <- str_c(path_res, "GenAM/Correlated/Gen_am_sampler/")
      }else if(sampler_res == "RAM"){ 
        sampler <- "RAM"
        path_files <- str_c(path_res, "RAM/Correlated/Ram_sampler/")
      }
      
      # Extract particle data 
      file_part_path <- str_c(path_files, "Pilot_runs/Exp_tag1/N_particles.tsv")
      n_part <- as.numeric(read_tsv(file_part_path, col_types = cols(), col_names = F)[1, 1])
      print(n_part)
      
      data_files <- list.files(path_files)
      data_files <- data_files[data_files != "Pilot_runs"]
      
      
      for(k in 1:length(data_files)){
        # Extract run 
        run <- as.integer(str_match(data_files[k], "run(\\d+).csv")[2])
        data <- as_tibble(fread(str_c(path_files, data_files[k]))) %>%
          select(-log_lik)
        n_samples <- dim(data)[1]
        data <- data[round(n_samples * 0.2):n_samples, ]
        multi_ess <- tryCatch(multiESS(data), error = function(e){return(0.0)})
        
        data_save <- tibble(multi_ess = multi_ess, run = run, sampler = sampler, start_guess = start_guess, 
                            data_set = data_set, n_part = n_part)
        
        data_res <- data_res %>%
          bind_rows(data_save)
        
      }
    }
  }
  
  return(data_res)
  
}


# Process single-individual functions 
model_name <- "Ornstein_test_proposal"
data_orn <- process_single_individual(model_name)

data_plot <- data_orn %>%
  mutate(sampler = as.factor(sampler)) %>%
  mutate(start_guess = as.factor(start_guess)) 


p <- ggplot(data_plot, aes(start_guess, multi_ess, fill = sampler)) + 
  geom_boxplot() + 
  geom_rangeframe(size = 1.0, sides = "l") + 
  scale_y_log10() + 
  geom_point(aes(color = n_part), position=position_jitterdodge(), size = 1.5) + 
  scale_fill_manual(values = my_colors[-1]) + 
  scale_color_viridis_c() + 
  labs(x = "", y = "") + 
  my_theme + theme(axis.text.x = element_blank(), legend.position = "none")
      
ggsave(str_c(dir_save, "Single_orn.svg"), p, bg = "transparent", width = BASE_WIDTH, height = BASE_HEIGHT)


# Process single-individual functions 
model_name <- "Schlogl_test_proposal"
data_sch <- process_single_individual(model_name)

data_plot <- data_sch %>%
  mutate(sampler = as.factor(sampler)) %>%
  mutate(start_guess = as.factor(start_guess)) %>%
  mutate(multi_ess = multi_ess + 1) %>%
  filter(multi_ess < 1e6) # Does not change results, but removes runs with bad inference runs (few outliers)


p <- ggplot(data_plot, aes(start_guess, multi_ess, fill = sampler)) + 
  geom_boxplot() + 
  geom_rangeframe(size = 1.0, sides = "l") + 
  scale_y_log10() + 
  geom_point(aes(color = n_part), position=position_jitterdodge(), size = 1.5) + 
  scale_fill_manual(values = my_colors[-c(1, 4)]) + 
  scale_color_viridis_c() + 
  labs(x = "", y = "") + 
  my_theme + theme(axis.text.x = element_blank(), legend.position = "none")
  
ggsave(str_c(dir_save, "Single_schlogl.svg"), p, bg = "transparent", width = BASE_WIDTH, height = BASE_HEIGHT)


p <- ggplot(data_plot, aes(start_guess, multi_ess, fill = sampler)) + 
  geom_boxplot() + 
  geom_rangeframe(size = 1.0, sides = "l") + 
  scale_y_log10() + 
  geom_point(aes(color = n_part), position=position_jitterdodge(), size = 2.5) + 
  scale_fill_manual(values = my_colors[-c(1, 4)]) + 
  scale_color_viridis_c() + 
  labs(x = "", y = "") + 
  my_theme + theme(axis.text.x = element_blank())

ggsave(str_c(dir_save, "Single_bar.svg"), p, bg = "transparent", width = BASE_WIDTH, height = BASE_HEIGHT)

