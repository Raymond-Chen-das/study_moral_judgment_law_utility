In this folder, you will find four data set files:

File 1: CountriesChangePr.csv
File 2: SharedResponse.csv
File 3: SharedResponsesFullFirstSessions.csv
File 4: SharedResponseSurvey.csv 

Please note the following keywords:

* Session: Each session consists of 13 scenarios (dilemmas) that are faced by an automated vehicle (AV). When a user visits the Judge mode, they respond to each dilemma by choosing one of the two outcomes (aka profiles). At the end of the 13 scenarios the user is presented with a summary of their decisions. At the end of the session users are also offered an optional survey (more on it below). Users are allowed to go through as many sessions as possible, and they may decide to leave the website or close the window before completing the session. This data is in the SharedResponse.csv file. When users opt-out from sharing their data by clicking the link "Opt-out", their data is removed from the database and a counter is incremented.

* Scenario: Each scenario resembles a dilemma faced by an AV. Each dilemma consists of two outcomes (aka profiles): one outcome is the result of the AV STAYing on course, and other is the result of the AV SWERVing off course. In the data set when a user chooses an outcome, this choice is logged in the database.

* Outcome: An outcome is represented by a set of features describing the environment and the characters whose lives are at stake in that outcome. While the data is collected and stored in the database at the level of the scenario (each record/row is a scenario), for the purpose of the analysis, the data is stored in the files File 2-4 at the level of the outcome. In other words, each scenario is represented by 2 rows in the data set files File 2-4. Each completed session is represented by 26 rows in the data sets, while non-completed sessions are represented by an even number of rows that's less than 26.

* Survey: At the end of the session users are shown a summary of their results and are also offered an optional survey to take. The order of showing the results vs. offering the survey is randomly allocated: half of the users are offered the survey before they see the results, while the other half is shown the results first, and then are asked to complete a survey. The survey contains demographic questions (age, gender, etc.) as well as other questions which are not included in the data sets.

* Factors/attributes: we manipulated the following 9 attributes: kind of intervention (stay/swerve), relationship to AV (pedestrians/passengers), legality (lawful/unlawful), gender (males/females), age (younger/older), social status (higher/lower), fitness (fit/large), number of characters (more/fewer), species (humans/pets). The first three attributes are structural attributes (relate to the environment), they are manipulated in every scenario. The remaining 6 attributes describe something about the characters (the potential casualties). Each of the 6 character attributes is manipulated independently in two scenarios per session (i.e. 12 scenarios in total). The 13th scenario is generate completely randomly. The order of these scenarios is randomised. Before you use this data set files, please refer to the Supplemental Information (SI) document in order to understand how these scenarios were generated.

* Characters: we considered 20 different characters who are characterised by some of the 6 character attributes. These characters are: man, woman, pregnant woman, baby stroller, elderly man, elderly woman, boy, girl, homeless person, large woman, large man, criminal, male executive, female executive, female athlete, male athlete, female doctor, male doctor, Dog, and Cat.


About the files:

File 1: CountriesChangePr.csv
=============================
The data in this file is calculated from SharedResponse.csv file. The file has 18 columns (2 x 9 attributes) and 130 rows (for 130 countries). This contains the effect sizes or the AMCEs (Estimate) and the user-level cluster-robust standard errors (se) of the effect sizes for 9 attributes. The Estimates have the same interpretation as in Figure 2: They represent the Average Marginal Causal Effect (AMCE) of each attribute (see the Supplementary Information document or Ref [18] from the main paper to understand more). For example, "Gender [Male -> Female]: Estimates" represents the change in probability (delta p) of saving the characters if we replace male characters with female characters (averaged over all other factors). The corresponding column "Gender [Male -> Female]: se" represents the (clustered) standard error of this estimate.

File 2: SharedResponse.csv
===========================
This contains the main data set that was used to produce Figure 2, the calculation of CountriesChangePr.csv file, and all user-level figures in the Extended Data Figures and the SI.

Before explaining the columns, see an example of one scenario in the data set file SharedResponse.csv. Recall that each scenario is represented by two rows in the data, each row is one of the two outcomes (note the matching ResponseID). An interpretation of the scenario represented in these two lines is provided below. 


 ResponseID            ExtendedSessionID       UserID ScenarioOrder Intervention PedPed
1: 2224g4ytARX4QT5rB 213978760_9992828917431898.0 9.992829e+15             7            0      0
2: 2224g4ytARX4QT5rB 213978760_9992828917431898.0 9.992829e+15             7            1      0
   Barrier CrossingSignal AttributeLevel ScenarioTypeStrict ScenarioType DefaultChoice
1:       1              0           Less        Utilitarian  Utilitarian          More
2:       0              1           More        Utilitarian  Utilitarian          More
   NonDefaultChoice DefaultChoiceIsOmission NumberOfCharacters DiffNumberOFCharacters Saved
1:             Less                       0                  4                      1     1
2:             Less                       0                  5                      1     0
   Template DescriptionShown LeftHand UserCountry3 Man Woman Pregnant Stroller OldMan OldWoman Boy
1:  Desktop                1        0          USA   0     0        0        1      0        0   0
2:  Desktop                1        1          USA 0.0     0        0        1      0        0   0
   Girl Homeless LargeWoman LargeMan Criminal MaleExecutive FemaleExecutive FemaleAthlete
1:    0        0          0        0        0             0               0             1
2:    0        0          0        0        0             0               0             1
   MaleAthlete FemaleDoctor MaleDoctor Dog Cat
1:           0            1          0   0   1
2:           1            1          0   0   1


This data set contains 41 columns. The last 20 columns represent the number of characters of each type in each outcome. For example, the columns "Man" and "Woman" represent the number of Man and Woman characters in each outcome (both are zero in this example). Note here that Man and Woman represent gendered (neutral-otherwise) adult characters. They are different from other gendered characters who has further descriptions like FemaleDoctor and LargeMan. In this dataset, the column Man provides the number of times the character Man exist in that outcome rather than the sum of all occurrences of male characters in that outcome.

With a quick look at this example you can see that one outcome (first row) contains a baby stroller, a female athlete, a female doctor, and a cat. The second outcome (second row) contains the same characters in addition to a male athlete. We have only one character of each type here, but each of these columns could take a value between 0 and 5, with the crucial restriction that the total number of characters in each outcome (row) is between 1 and 5. This number is captured in the column "NumberOfCharacters". The column "DiffNumberOFCharacters" captures the absolute difference in total number of characters between the two outcomes, which is 1 in this case.

Now, we visit the remaining 21 columns which are available in all three files:

['ResponseID', 'ExtendedSessionID', 'UserID', 'ScenarioOrder',
 'Intervention', 'PedPed', 'Barrier', 'CrossingSignal', 'AttributeLevel',
 'ScenarioTypeStrict', 'ScenarioType', 'DefaultChoice', 'NonDefaultChoice', 'DefaultChoiceIsOmission', 
'NumberOfCharacters', 'DiffNumberOFCharacters', 'Saved', 
'Template', 'DescriptionShown', 'LeftHand', 'UserCountry3']

- ResponseID: a unique, random set of characters that represents an identifier of the scenario. Since each scenario is represented by 2 rows, every row should share a 'ResponseID' with another row. For the purpose of analysis there was some filtering early on at the level of outcomes (see code files). As such, it is possible that some rows in the data set SharedResponse.csv do not share a RespondID with any other row. If a scenario-level data set is desired (each row represents a scenario), this column can be used to reshape the data set by matching two rows of the same ResponseID.

- ExtendedSessionID: a unique, random set of characters that represents an identifier of the session. This ID combines a randomly generated ID for the session, concatenated with the UserID. 

- UserID: a unique, random set of characters that represents an identifier of the user (respondent), captured using browser fingerprints.

- ScenarioOrder: this takes a value between 1 and 13, representing the order in which the scenario was presented in the session.

- Intervention: represents the decision of the AV (STAY or SWERVE) that would lead to this outcome [0: the character would die if the AV stays, 1: the character would die if AV swerves]. This is not the actual decision taken by the user, but rather a part of the structural characterisation of the scenario.

- PedPed: every scenario has either pedestrians vs. pedestrians or pedestrians vs. passengers (or passengers vs. pedestrians). This column provides information about not just this outcome, but about the combination of both outcomes in the scenario; whether the scenario pits pedestrians against each other or not [1: pedestrians vs. pedestrians, 0: pedestrians vs. passengers (or vice versa)]

- Barrier: Another structural column which describes whether the potential casualties in this outcome are passengers or pedestrians [1: passengers, 0: pedestrians]. This column was used to calculate PedPed (after matching rows on RespondID).

- CrossingSignal: Another structural column which represents whether there is a traffic light in this outcome, and light colour if yes [0: no legality involved, 1: green or legally crossing, 2: red or illegally crossing]. Every scenario that has pedestrians vs. pedestrians (i.e. PedPed=1) features one of three legality-relevant characterisations: a) the pedestrians on both sides are crossing with no legal complications, b) one group is crossing legally (on a green light), while the other is crossing illegally (on a red light), and c) vice versa. Every scenario that has pedestrians vs. passengers (i.e. PedPed=0) features also one of three legality-relevant characterisations: a) the pedestrians are crossing with no legal complications, b) the pedestrians are crossing legally (on a green light), and c) pedestrians are crossing illegally (on a red light). There are no legality concerns for passengers.

- Saved: this resembles the actual decision made by the user [1: user decided to save the characters in this outcome, 0: user decided to kill the characters in this outcome]. Note that this column is reverse coded from the database. On the website, users click on the outcome they choose. That means the choice they make results in the death of the characters represented in that outcome (with a skull sign on the website). You can imagine another column named "Killed" which would be the exact opposite of "Saved" (i.e. 1 if Saved is 0 and 0 if Saved is 1).

- NumberOfCharacters: takes a value between 1 and 5, the total number of characters in this outcome. This is the sum of numbers in the last 20 columns (character columns). It also represents the number of characters who will be saved or killed based on “Saved" value.

- DiffNumberOFCharacters: takes a value between 0 and 4; difference in number of characters between this outcome and the other outcome.

- Template: the scenario was answered using 'desktop', 'mobile', or 'tablet'.

- DescriptionShown: whether the user clicked “show description” to present the textual description of the scenario before choosing the outcome [1: description button was clicked at least once for this outcome, 0 description button was never clicked for this button].

- LeftHand: The order of the two outcomes (as a result of STAY vs. SWERVE decision) is randomised. This column captures the position of the outcome as presented to the users [1: this outcome appeared on the lefthand side of screen, 0: this outcome appeared on the righthand side of the screen].

- UserCountry3: the alpha-3 ISO code of the country from which the user accessed the website. This is generated from the user IP which is collected but not shared here.

- ScenarioType and ScenarioTypeStrict: These two columns have 7 values, corresponding to 7 types of scenarios (6 attributes + random). These are: "Utilitarian","Gender", "Fitness", "Age", "Social Value", "Species", and "Random".
In the early stage of the website, we forgot to include a code that gives the scenario type (one of the 6 categories mentioned above + random). We had to write a code to figure that out from the character types. This is the "ScenarioType" column. Some scenarios who were generated as part of the "random", could fit in one of the 6 other categories. Later, we used a clear parameter to capture this type, which is in "ScenarioTypeStrict". Thus, this column provides an accurate description, but it does not have a value for the early scenarios. In the analysis for the figures, whenever we filtered based on the scenario type, we used both columns. For example, to filter the age related scenarios, we use:
	ScenarioTypeStrict=“Age” && ScenarioType=“Age” 
where "&&” is the logic AND.

- AttributeLevel: is dependent on the scenario type. Each scenario type (except random) has two levels: 
+Gender: [Males: characters are males, Females: characters are females]
+Age: [Young: characters in this outcome are younger (Boy/Girl + Man/Woman) than in the other outcome, Old: characters in this outcome are older (Elderly Man/Woman and Man/Woman)].
+Fitness: [Fit: characters in this outcome are more fit (Male/Female Athlete and Man/Woman), Fat: characters in this outcome are less fit (Large Man/Woman and Man/Woman)].
+Social Value: this was changed in the analysis to "social status" instead, and the characters Male/Female Doctor and Criminal were filtered out [High: characters in this outcome have higher social status (Male/Female Executives and Man/Woman), Low: characters have a lower social status (Homeless and Man/Woman)]
+Species: [Hoomans: characters in this outcome are humans (all but Dog/Cat), Pets: characters in this side are pets (Dog/Cat)]
+Utilitarian: [More: there are more characters in this outcome, Less: there are fewer people in this outcome]. In fact, the characters on the "More" side are the same characters on the "Less" side, in addition to at least one more characters. (excuse the error in using "Less" for a countable)
+ Random: it has one value ["Rand": characters in both outcomes are randomly generated].


- DefaultChoice', 'NonDefaultChoice':
Default Choice depends on the Scenario Type. This was chosen randomly (the word “default" here has no special meaning). For the following Scenario Types: ["Gender", "Fitness", "Age", "Social Status", “Species”, "Utilitarian"], the default choice is ["Male", "Fit", "Young", "High", “Hoomans”, “More"], while the non-default choice is ["Female", "Fat", "Old", "Low", “Pets”, ”Less"]. These two columns are only useful as an aid for the following column, "DefaultChoiceIsOmission".

- DefaultChoiceIsOmission: Omission here means no intervention (Intervention =0), and default choice is as described from "DefaultChoice" column. When DefaultChoiceIsOmission = 1 it means that based on the scenario type, characters that hold the default choice (that is, males for gender, fit for fitness, young for age,…etc.) will be the ones killed if the AV does nothing (omission or no intervention). On the other hand, if DefaultChoiceIsOmission = 0, then the characters that hold the non-default choice will be the ones killed if the AV does nothing (i.e., the characters holding the default choice will be the ones killed if the car swerves (i.e., intervenes)).

For example, suppose that the "ScenarioType" is "Gender". This means that "DefaultChoice" = "Males". When the column "DefaultChoiceIsOmission" is equal to 1 it means that Males will die on the omission decision (i.e. if the AV stays on its route), while Females will die on the commission (i.e. if the AV swerves). When "ScenarioType" = "Age", '"DefaultChoiceIsOmission"=1' means younger people will die on the omission, etc.

After explaining the columns, now we get back to the example scenario above. This scenario represents the dilemma in which there are 4 passengers in the AV, and 5 pedestrians who are crossing legally. There is a barrier in front of the AV. If the AV stays, it will hit the barrier and kill the 4 passengers, if it swerves it will kill the 5 pedestrians. The user chose for the AV to SWERVE and thus has decided to sacrifice the 5 pedestrians and spare the 4 passengers.


3- SharedResponsesFullFirstSessions.csv
=======================================
This file contains responses from only the first completed session by each user. Recall from above that users can drop mid session and can also answer multiple sessions. This data set only contains one completed session by each user (that's the first one they completed). This data set does not contain all columns from SharedResponse.csv the 20 columns representing the number of characters of each type are not included. The data from this data set is only used for Extended Data Figure 2 in the paper.

4- SharedResponseSurvey.csv
============================ 
This file contains data from SharedResponse.csv but only for users who completed the optional 'Survey' about themselves. Since this survey only appears after completion, this means that each session in this data set is a complete session, but multiple sessions could be available for the same user. It also means that each session is represented by 26 rows in the data set. Note here that the demographic data was (inner) merged/joined with SharedResponse.csv data (based on ExtendedSessionID), which means that each demographic value is repeated 26 times in this data set. This data set was used for every figures or analysis that includes demographics (including Extended Data Figure 2 in the paper).

This file contains the 21 columns from above in addition to the following columns, which represent survey data, corresponding to respondent's provided data about their demographics: 

- Review_age: answer to the question "How old are you?" Age is collected via an empty box (users can input numbers). Only data corresponding to age between 18 and 75 is included.

- Review_education: answer to the question "Highest level of education". The following options are presented (in the English version): "Less Than a High School Diploma" (coded: underHigh), "High School Diploma" (high), "Vocational Training" (vocational), "Attended College" (college), "Bachelor Degree" (bachelor), "Graduate Degree" (graduate), "Other" (other), no answer (default).

- Review_gender: answer to the question "What is your gender?" The following options are presented (in the English version): "Male" (male), "Female" (female), "Other" (other), no answer (default).

- Review_income: answer to the prompt "Annual income, including tips, dividends, interest, etc. (in US dollars)". The following options are presented (in the English version): "
"Under $5,000" (coded under5000),"$5,000-$10,000" (5000), "$10,001-$15,000" (10000), "$15,001-$25,000" (15000), "$25,001-$35,000" (25000), "$35,001-$50,000" (35000), "$50,001-$80,000" (50000), "$80,001-$100,000" (80000), "Over $100,000" (above100000), no answer (default).

- Review_political: A slider from Conservative (0) to Progressive (1) is presented to users. The value captured correspond to a number between 0 and 1. Default (no answer) is 0.5

- Review_religious: A slider from Not Religious (0) to Very Religious (1) is presented to users. The value captured correspond to a number between 0 and 1. Default (no answer) is 0.5