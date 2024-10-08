### Dataset 1: Credit-G

#### Description
This dataset classifies individuals as "good" or "bad" credit risks based on various financial and personal characteristics. The target variable is labeled as **class** with two categories: **good** (good credit) and **bad** (bad credit).

#### Overview
- **Source:** German Credit dataset
- **Purpose:** To classify individuals' credit risk, featuring a cost matrix that indicates the relative importance of classification errors (e.g., misclassifying a bad risk as good incurs higher costs).

#### Dataset Characteristics
- **Number of Rows:** 1,000
  - Each row represents a person with financial and personal attributes.
- **Number of Columns:** 21 (including the target variable)
- **Data Types:**
  - 14 categorical columns
  - 7 integer columns

#### Key Features
- **checking_status:** Status of the person's checking account.
- **credit_history:** History of credit payments (e.g., no delays, critical accounts).
- **duration:** Duration of the credit (in months).
- **age:** The person's age.

The **class** column indicates whether the credit risk is "good" or "bad."

---

### Dataset 2: Mushroom

#### Description
This dataset classifies mushrooms as either "edible" (e) or "poisonous" (p) based on their physical characteristics. The target variable is labeled as **class** with two categories: **e** (edible) and **p** (poisonous).

#### Overview
- **Purpose:** To classify mushrooms using attributes such as color, odor, and habitat.

#### Dataset Characteristics
- **Number of Rows:** 8,124
  - Each row represents a mushroom with various physical characteristics.
- **Number of Columns:** 23 (including the target variable)
- **Data Types:** All columns are categorical.

#### Key Features
- **cap-shape:** Shape of the mushroom cap (e.g., bell-shaped, flat).
- **odor:** Smell of the mushroom (e.g., almond, foul).
- **gill-color:** Color of the mushroom's gills.

The **class** column indicates whether a mushroom is "edible" or "poisonous."
