
At the start of our project, we began by collecting data. From NILU we downloaded CSV
files containing air quality data, while weather data from Meteorologisk Institutt was
gathered through an API in the form of a JSON file. Although this worked initially, we
realized not all group members could access the data simultaneously, as it was stored
in a folder on one member’s desktop. To resolve this, we sought help to set up a virtual
environment, allowing shared access to a common file location. Towards the end of the
project, we also found that managing large datasets by manually downloading CSV files
could be demanding in terms of time and be quite tedious, making the use of an API a
more practical and scalable choice.

Another recurring challenge was the use of branches. Despite weekly planning and
communication, we often ended up working on the same files, leading to overwrites and
disorganized code. Our initial approach was to notify each other when committing
changes, and assign separate files to work on for each developer. A few weeks in, we
adopted the use of branches, and quickly realized it was a much better solution. We
added restrictions to GitHub to prevent direct commits to the main branch and
established a weekly merge routine to keep the main code updated before creating new
branches.

During the weekly meetings we divided the tasks between us and created a branch for
each developer and the task they would be working on. This taught us the value of
patience, teamwork, and maintaining a clear project structure. At first we didn’t consider
the commit messages and summaries, and kept the automatically generated ones.
Once we were working with branches, and begane using tags, we saw the value in
descriptive commit messages and summaries. While not always remembered, we
began writing the commit messages ourselves to more easily keep track of the
changes. For future projects, we’ll make sure we write informative commit messages
and summaries, and create branches dedicated to specific tasks.

Another challenge was handling data discrepancies. Missing values and poor quality
data was replaced with NaN for the calculations and graph designs, to exclude them.
Once we got to predictive analysis it became necessary to include interpolation in the
process. As the project progressed, program runtimes increased, especially with large
datasets. To manage this, we created a main function to call data-handling functions
from other folders, organizing the code into smaller sections. This improved both
structure and readability. We also used lists instead of repeatedly generating
DataFrames, reducing runtime significantly. We learned how to implement unit tests,
which, although time-consuming, proved useful for debugging and ensuring function
accuracy.

For data visualization, we used graphs with dual y-axes to compare two environmental
datasets. Libraries such as Seaborn and NumPy helped in this process. A key issue
was inconsistent y-axis formatting, which we resolved by using the twinx() function in
Matplotlib to generate double-sided graphs. Fixing one problem often introduced new
ones, so we chose to compromise on minor visual flaws in favour of more functional
solutions, ensuring that visualizations showed general trends effectively. We grouped
various types of air pollution data on the same graph and used color coding for clarity.
For regression curves, we mainly used third- or fourth-degree polynomials. While these
capture general trends, accuracy could be improved with higher-degree polynomials or
locally weighted regression, which focuses more on local data points.

We also learned that visualization helped us identify anomalies that could distort
regression models. Initially, we planned to use standard outlier formulas and treat
outliers as missing data, but this removed too much valid data. Eventually, we defined
outliers as values outside three standard deviations from the mean, but left the “3” as a
variable that can be changed to better fit future data that might be used.

This project gave us valuable insight into how programming can be used in predictive
analysis. Despite some inconsistencies, we gained a better understanding of how to
process and adapt data for future predictions. We also discovered interesting
correlations, such as between temperature and nitrogen oxide, or rainfall and air
particles. Additionally, we developed a stronger grasp of statistical methods, especially
in handling missing data, outliers, and anomalies, which could be great tool in future
work life while dealing with large data sets. The project's execution encouraged critical
thinking and open-mindedness as we explored suitable statistical tools for our scenario.
We became familiar with techniques for calculating averages and standard deviation,
creating box plots, using APIs, and building regression models. This project has helped
us develop project management skills that will be applicable in many future work
situations. While not every developer will use programming in daily work, the general
knowledge of data handling and environmental processes can have many uses in future
projects, work situations, and daily life.

We also believe that this project is a great starting point for further analysis of data. One
interesting field of research could be to use more restrictions and regulations set by the
government, or whether the obs