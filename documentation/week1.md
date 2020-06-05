### Week #1
#### Collins Nji

This week, was my first week at CSET! It started with an introduction to the project I would be working on for the next couple of weeks, followed by some preliminary reading. Jennifer walked me through a lot of what I did this week, starting with the project outline. 

I started by reading some WIPO articles on vehicle/driver detection and identification, noting down some key terms related to that field. Below are some of the keywords for Vehicle/Driver Recognition I could think of
Connected / Autonomous Vehicles
Automated driving systems
Vehicle / Driver recognition
Recognition

I then wrote a query on the NormalizedNames column in the FieldsOfStudy table trying out various keywords that I came up with. Queries that include the words “Connected or Autonomous Vehicle” tend to discuss problems AVs face, rather than what we are interested in finding: Driver/Vehicle Identification.

I started by running the following query:
```sql
SELECT distinct(NormalizedName) FROM `***REMOVED***.gcp_cset_mag.FieldsOfStudy`
where NormalizedName like '%{{relevant_term}}%'
```
```
* with the following relevant_terms:
	vehicle, recognition, detection
```
There were many fields that discussed vehicle types, but not a lot about recognition or identification. However, looking at the results, I could gather that the following fields seemed relevant: 
- vehicle identification
- automatic vehicle identification
- automatic vehicle location
- vehicle detection
- vehicle detector
- driver recognition
- road sign recognition

It is worth noting that in this context, vehicle detection, and vehicle recognition are considered the same thing. Hence they are both relevant. Following suggestions from Jennifer, I ran the following queries to generate three different data samples and combined them into a single file (merged_data.csv)

1. Select papers that are likely relevant
```sql
select paper.PaperId, paper.PaperTitle, paper.Abstract, RAND() as rand
from gcp_cset_mag.PapersWithAbstracts paper
left join gcp_cset_mag.PaperFieldsOfStudy paper_fos on paper.PaperId = paper_fos.PaperId
left join gcp_cset_mag.FieldsOfStudy fieldOfStudy on paper_fos.FieldOfStudyId = fieldOfStudy.FieldOfStudyId
where regexp_contains(fieldOfStudy.NormalizedName, r"(?i)vehicle.*identification$|driving.*recognition|vehicle.*detection|vehicle detector")
and (DocType != "Dataset") and (DocType != "Patent")
order by rand desc
limit 80
```
2. Select confusable papers (papers that only slightly related to driver/vehicle identification)
```sql
select paper.PaperId, paper.PaperTitle, paper.Abstract, RAND() as rand
from gcp_cset_mag.PapersWithAbstracts paper
left join gcp_cset_mag.PaperFieldsOfStudy paper_fos on paper.PaperId = paper_fos.PaperId
left join gcp_cset_mag.FieldsOfStudy fieldOfStudy on paper_fos.FieldOfStudyId = fieldOfStudy.FieldOfStudyId
where regexp_contains(fieldOfStudy.NormalizedName, r"(?i)computer vision")
and (DocType != "Dataset") and (DocType != "Patent")
order by rand desc
limit 60
```
3. Select papers that are likely not relevant at all
```sql
select paper.PaperId, paper.PaperTitle, paper.Abstract, RAND() as rand
from gcp_cset_mag.PapersWithAbstracts paper
left join gcp_cset_mag.PaperFieldsOfStudy paper_fos on paper.PaperId = paper_fos.PaperId
left join gcp_cset_mag.FieldsOfStudy fieldOfStudy on paper_fos.FieldOfStudyId = fieldOfStudy.FieldOfStudyId
where not regexp_contains(fieldOfStudy.NormalizedName, r"(?i)vehicle.*identification$|driving.*recognition|vehicle.*detection|vehicle detector|computer vision")
and (DocType != "Dataset") and (DocType != "Patent")
order by rand desc
limit 60
```
From here, I merged the three CSV files I got from running the queries with
```
sed 1d *.csv > merged_data.csv
```
After merging the CSV files, I manually annotated the papers by adding a Relevancy column to the merged data. If the paper’s abstract described a vehicle/driver detection or identification, I marked it as relevant. Otherwise, I marked it as irrelevant. It is worth noting that in the random sample I obtained above, there were a lot of vehicle detection papers and not so much driver identification. In cases where the paper abstract was missing (ie. null) I used the paper title to try to determine if the paper was relevant or not. If the paper was in a language other than English, I marked it as irrelevant. In the file `no_abstract.txt` I listed all the Paper IDs that where either missing an abstract or were not in English

After I annotating the merged CSV file, I extracted the Paper ID and the Relevancy column into a separate data file. 

```python
import pandas as pd
df = pd.read_csv("merged_data.csv", prefix=None)
data = df[['PaperId', 'Relevancy']]
data.to_csv('relevant_mags.csv', encoding='utf-8', index=False)
```

All data files are in the `week1-data` directory 