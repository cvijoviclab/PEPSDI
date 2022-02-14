library(dplyr)
library(readr)
library(stringr)
library(transport)
library(data.table)


read_data <- function(model_name)
{
  dir_data <- str_c("../../../Intermediate/Single_individual/", model_name)
  
  data_chain_save <- tibble()
  # Remove Kalman filter benchmarks 
  file_list <- list.files(dir_data)
  file_list <- file_list[file_list != "Kalman_1"]
  file_list <- file_list[file_list != "Kalman_12"]
  file_list <- file_list[file_list != "Kalman_123"]
  file_list <- file_list[file_list != "Distance"]
  
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
      
      # Aggregate chains 
      for(k in 1:length(data_files)){
        # Extract run 
        run <- as.integer(str_match(data_files[k], "run(\\d+).csv")[2])
        data <- as_tibble(fread(str_c(path_files, data_files[k]))) %>%
          select(-log_lik)
        n_samples <- dim(data)[1]
        data <- data[round(n_samples * 0.2):n_samples, ]
        data_chain <- data %>%
          mutate(run = run, sampler = sampler, start_guess = start_guess, data_set = data_set, n_part = n_part)
        data_chain_save <- data_chain_save %>%
          bind_rows(data_chain)
      }
    }
  }
  
  return(data_chain_save)
}


compute_distance <- function(data_chain, n_cores=1)
{

  sampler_list <- c("RAM", "AM", "GenAM")
  sg_list <- 1:5
  data_set_list <- c(1, 12, 123)
  run_list <- 1:10
  
  print("Starting distance calculations")
  list_res <- parallel::mclapply(run_list, function(i){
    
    run_use <- i
    result <- tibble()
    
    for(j in 1:length(data_set_list)){
      for(k in 1:length(sg_list)){
        for(l in 1:length(sampler_list)){
          
          data_set_use <- data_set_list[j]
          data_set_c <- as.character(data_set_list[j])
          sg_use <- sg_list[k] 
          sampler_use <- sampler_list[l]
          
          data_val <- read_csv(str_c("../../../Intermediate/Single_individual/Ornstein_test_proposal/Kalman_", data_set_c, "/Kalman.csv"), col_types = cols())
          data_test <- data_chain %>%
            filter(run == run_use & start_guess == sg_use & sampler == sampler_use & data_set == data_set_use)
          
          data_val_use <- pp(as.matrix(data_val[33000:48000, 1:3]))
          data_test_use <- pp(as.matrix(data_test[33000:48000, 1:3]))
          
          dist <- wasserstein(data_val_use, data_test_use, p=1)
          
          result_tmp <- tibble(dist = dist, sampler = sampler_use, start_guess = sg_use, data_set = data_set_use, run = run_use)
          result <- result %>% bind_rows(result_tmp)
        }
      }
    }
    
    return(result)}, mc.cores = n_cores)
  
  data_distance <- do.call(rbind, list_res)
  return(data_distance)
}


model_name <- "Ornstein_test_proposal"
data_chain <- read_data(model_name)
data_dist <- compute_distance(data_chain, n_cores = 10)

# Save expansive distance calculations to disk 
dir_save <- "../../../Intermediate/Single_individual/Ornstein_test_proposal/Distance/"
if(!dir.exists(dir_save)) dir.create(dir_save, recursive = T)
write_csv(data_dist, str_c(dir_save, "Distance_data.csv"))
