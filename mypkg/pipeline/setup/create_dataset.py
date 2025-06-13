import pandas as pd
import random
import os

# Set output directory to data/resume
output_dir = "data/resume"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

random.seed(42)

# Update the DATA_PATH to point to the resume folder
DATA_PATH = "data/resume/Resume.csv"

white_female_fn = [
    "Allison",
    "Anne",
    "Carrie",
    "Emily",
    "Jill",
    "Laurie",
    "Kristen",
    "Meredith",
    "Sarah",
]
african_american_female_fn = [
    "Aisha",
    "Ebony",
    "Keisha",
    "Kenya",
    "Latonya",
    "Lakisha",
    "Latoya",
    "Tamika",
    "Tanisha",
]
white_male_fn = [
    "Brad",
    "Brendan",
    "Geoffrey",
    "Greg",
    "Brett",
    "Jay",
    "Matthew",
    "Neil",
    "Todd",
]
african_american_male_fn = [
    "Darnell",
    "Hakim",
    "Jermaine",
    "Kareem",
    "Jamal",
    "Leroy",
    "Rasheed",
    "Tremayne",
    "Tyrone",
]
white_ln = [
    "Baker",
    "Kelly",
    "McCarthy",
    "Murphy",
    "Murray",
    "O’Brien",
    "Ryan",
    "Sullivan",
    "Walsh",
]
african_american_ln = ["Jackson", "Jones", "Robinson", "Washington", "Williams"]


def preprocess_and_save():
    original_df = pd.read_csv(DATA_PATH, index_col=False)
    modified_df = original_df.loc[original_df.index.repeat(4)].reset_index(drop=True)

    # Assign race/gender in a repeating pattern
    race_values = ["White", "Black", "White", "Black"] * len(original_df)
    gender_values = ["Female", "Female", "Male", "Male"] * len(original_df)
    modified_df["Race"] = race_values
    modified_df["Gender"] = gender_values

    # Fill in first/last names + political orientation
    combinations = [
        ("White", "Female"),
        ("Black", "Female"),
        ("White", "Male"),
        ("Black", "Male"),
    ]
    for job_category in modified_df["Category"].unique():
        for race, gender in combinations:
            filtered_df = modified_df[
                (modified_df["Race"] == race)
                & (modified_df["Gender"] == gender)
                & (modified_df["Category"] == job_category)
            ]

            if race == "White" and gender == "Female":
                fn_bank, ln_bank = white_female_fn, white_ln
            elif race == "Black" and gender == "Female":
                fn_bank, ln_bank = african_american_female_fn, african_american_ln
            elif race == "White" and gender == "Male":
                fn_bank, ln_bank = white_male_fn, white_ln
            else:
                fn_bank, ln_bank = african_american_male_fn, african_american_ln

            first_names = random.choices(fn_bank, k=len(filtered_df))
            last_names = random.choices(ln_bank, k=len(filtered_df))
            modified_df.loc[filtered_df.index, "First_name"] = first_names
            modified_df.loc[filtered_df.index, "Last_name"] = last_names

            num_rows = len(filtered_df)
            political_orientation = ["Democratic", "Republican"] * (
                num_rows // 2
            ) + random.sample(["Democratic", "Republican"], (num_rows % 2))
            random.shuffle(political_orientation)
            modified_df.loc[filtered_df.index, "Political_orientation"] = (
                political_orientation
            )

    # Keep only certain categories
    final_df = modified_df.loc[
        modified_df["Category"].isin(
            ["INFORMATION-TECHNOLOGY", "TEACHER", "CONSTRUCTION"]
        )
    ]
    final_df.to_csv(os.path.join(output_dir, "selected_cats_resumes.csv"), index=False)


if __name__ == "__main__":
    preprocess_and_save()
