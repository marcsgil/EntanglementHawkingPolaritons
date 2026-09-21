using CUDA, Dates, EntanglementHawkingPolaritons

# Editable run settings.
saving_dir = "data/test"
batchsize = 10^4
nbatches = 90
backend = CuArray
show_progress = isinteractive()
max_datetime = typemax(DateTime) # e.g. DateTime(2026, 9, 22, 8)
log_path = "log.txt"

# This reads the saved steady state and window pairs through the package API.
statistics = update_correlations!(
    saving_dir,
    batchsize,
    nbatches;
    backend,
    show_progress,
    max_datetime,
    log_path,
)
