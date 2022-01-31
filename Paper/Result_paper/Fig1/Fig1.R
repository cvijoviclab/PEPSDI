library(tidyverse)
library(ggthemes)
library(latex2exp)
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

dir_save <- "../../../Results/Paper_figures/Fig1/"
if(!dir.exists(dir_save)) dir.create(dir_save)


path_data <- "../../../Intermediate/Simulated_data/SSA/Multiple_ind/Clock_fig1/Clock_fig1.csv"
data_plot <- read_csv(path_data, col_types = cols()) %>%
  mutate(id = as.factor(id))

range_data <- tibble(x = c(0.0, 72), y = c(0, 600))
p <- ggplot(data_plot, aes(time, obs)) + 
  geom_line(aes(time, obs, group=id), color = my_colors[1], size = 0.2) +
  geom_rangeframe(data=range_data, mapping=aes(x, y), color = "black", size = 1.0) + 
  labs(x = "", y = "") + 
  my_theme
ggsave(str_c(dir_save, "Data_ex.svg"), p, bg = "transparent", width = BASE_WIDTH, height = BASE_HEIGHT)
