
# %%
from qufi import read_results_directory, generate_all_statistics
results = read_results_directory("./tmp/")

# %%
generate_all_statistics(results)

# %%
