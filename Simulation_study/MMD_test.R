

# install.packages("Ecume")
# install.packages("Peacock.test")

library(Ecume)
library(rstudioapi)
library(Peacock.test)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
getwd()

ind = 5

n_sample = as.matrix(read.csv(paste0("n_sample_", ind, ".csv")))
t_sample = as.matrix(read.csv(paste0("t_sample_", ind, ".csv")))
gAE_recon = as.matrix(read.csv(paste0("gAE_recon_", ind, ".csv")))
VAE_recon = as.matrix(read.csv(paste0("VAE_recon_", ind, ".csv")))
gAE_gen = as.matrix(read.csv(paste0("gAE_gen_", ind, ".csv")))
VAE_gen = as.matrix(read.csv(paste0("VAE_gen_", ind, ".csv")))


mmd_test(t_sample, n_sample, type = "unbiased")
mmd_test(t_sample, gAE_recon, type = "unbiased")
mmd_test(t_sample, VAE_recon, type = "unbiased")
mmd_test(t_sample, gAE_gen, type = "unbiased")
mmd_test(t_sample, VAE_gen, type = "unbiased")

mmd_test(t_sample, n_sample, type = "linear")
mmd_test(t_sample, gAE_recon, type = "linear")
mmd_test(t_sample, VAE_recon, type = "linear")
mmd_test(t_sample, gAE_gen, type = "linear")
mmd_test(t_sample, VAE_gen, type = "linear")

# peacock2 : two-dimensional Kolmogorov-Smirnov test
peacock2(t_sample, n_sample)
peacock2(t_sample, gAE_recon)
peacock2(t_sample, VAE_recon)
peacock2(t_sample, gAE_gen)
peacock2(t_sample, VAE_gen)
