# Project Instructions

Write clear, concise, and readable code.

Use Google-style formatting:

- Use clear function and variable names.
- Add comments or docstrings where they help explain non-obvious logic.
- Add simple comments only when they help explain the logic.

Keep solutions simple. This is not production code, so avoid extensive error
handling, logging, config files, or unnecessary abstractions.

Do not create new code files unless specifically asked to do so, or unless you have asked the user and received permission.

When possible, display results clearly using pandas DataFrames or other simple output methods. 
When you create figures or results, save them to `./results/`.
Call `plt.show()` only after saving.
When possible, compute basic statistics and visualize them with plots.
The plots should have large text and number should to be formatted with :, .

Keep the code easy to understand, modify, and reuse.

In Jupyter Notebook (`.ipynb`) files, cells should follow a consistent three-step structure:

1. **Function Definition** – Define a function that performs a specific task.
2. **Function Call** – Call the function to execute the task.
3. **Results and Visualization** – Display the results, including relevant statistics, tables, and plots.

**Example:**
- **Cell 1:** Define a function to load the dataset.
- **Cell 2:** Call the function to load the data.
- **Cell 3:** Display descriptive statistics and visualizations of the loaded data.

Repeat this structure for each logical step in the notebook to ensure clarity, readability, and consistency.

Use Markdown headings at different levels (#, ##, ###, etc.) to create a clear, logical, and hierarchical structure throughout the Jupyter Notebook (.ipynb).

When downloading a new package, add it to `requirements.txt` with its version.

For each function make sure to have google-style description.