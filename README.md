# exploratory-data-analysis---customer-loans-in-finance271

The aim of the project is to perform Exploratory Data Analysis (EDA) on loans_payment database and gain deeper understanding of the loans data by identifying any patterns that could exist, identifying and correcting issues found (such as missing or incorrectly formatted data), applying statistical techniques to gain insight on the data's distribution, apply visualisation techniques to identify patterns and trends and finally present the results in a understandable and concise manner. 

<ins>Installation Instructions</ins>: Download the loan_payments database from the remote AWS RDS server using the credentials.yaml (file stored locally and ignored by GitHub) and running commands written in the db_utils.py file and then store the downloaded database locally under the name "loan_payments.csv" for easy access. Then another five files called "class_data_frame_info.py", "class_data_transform.py", "class_data_frame_transform.py", "class_plotter.py" and "project_final.ipynb" should be downloaded locally in the same directory as "loan_payments.csv" file. 

<ins>Usage Instructions</ins>: Open the "project_final" notebook. In the notebook, run the code that will read and load "loan_payments" data and import the four classes from four different Python files ("class_data_frame_info.py", "class_data_transform.py", "class_data_frame_transform.py" and "class_plotter.py"). Afterwards just read the text and run the code in order. The text and code in Milestone 2 downloads and saves locally the loans_payment data, in Milestone 3, the loans_payment data is cleaned and inspected for skewness, outliers and correlation and then in Milestone 4, loans_payment data is used to analyse and visualise current state of the loans, calculate the loss, projected loss and possible loss and to search for indicators of loss. 

**File Structure of the Project:**
 - *db_utils.py:* contains load_credentials_file() function which reads the locally stored credentials.yaml file and class RDSDatabaseConnector which uses the information in credentials.yaml to connect to the cloud database and extract the "loans_payment" data for viewing.
 - *.gitignore:* contains instructions to ignore credentials.yaml file (and other chosen files and file types) to be uploaded or stored on Github (due to security reasons).
 - *class_data_frame_info.py:* contains class DataFrameInfo which is imported to "project_final" notebook. 
 - *class_data_transform.py:* contains class DataTransform which is imported to "project_final" notebook.
 - *class_data_frame_transform.py:* contains class DataFrameTransform which is imported to "project_final" notebook.
 - *class_plotter.py:* contains class Plotter which is imported to "project_final" notebook.
 - *project_final.ipynb:* the notebook that loads and reads the "loan_payment.csv" file, imports all four classes from the separate Python files and uses this to clean and analyse loans payments dataset. </ul>
 
<ins>License Information</ins>: No License.


