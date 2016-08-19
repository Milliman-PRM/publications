/*
### CODE OWNERS: Ben Copeland

### OBJECTIVE:
	Pull a sample of member data for case-study purposes

### DEVELOPER NOTES:
	Members will have to be manually selected based on what you're looking for
*/
options sasautos = ("S:\Misc\_IndyMacros\Code\General Routines" sasautos) compress = yes;
%include "%sysget(UserProfile)\HealthBI_LocalData\Supp01_Parser.sas" / source2;
%include "&M073_Cde.PUDD_Methods\*.sas";

/* Libnames */
libname M160_Out "&M160_Out.";

/*Create some subsets of the feature dataset for members with certain conditions*/
data diab_members;
	set M160_Out.cond_cred_tune_feat;
	where
		condition_name eq 'Diabetes mellitus with complications- Chronic'
		and sqrt_adjusted_credibility gt 1
	;
run;

data heart_members;
	set M160_Out.cond_cred_tune_feat;
	where
		condition_name eq 'Coronary atherosclerosis and other heart disease- Chronic'
		and sqrt_adjusted_credibility gt 1
	;
run;

proc sql;
	create table diab_members_lim
	as select
		src.*
	from M160_Out.cond_cred_tune_feat as src
	left join diab_members as lim on
		src.member_id eq lim.member_id
	where lim.member_id is not null
	;
quit;
proc sql;
	create table heart_members_lim
	as select
		src.*
	from M160_Out.cond_cred_tune_feat as src
	left join heart_members as lim on
		src.member_id eq lim.member_id
	where lim.member_id is not null
	;
quit;

/*Select a few interesting members and put them into a list below*/
%let interesting_mems = 'MEMBER1','MEMBER2';

data predictions_lim;
	set M160_Out.preds;

	where member_id in (
		&interesting_mems.
		);
run;

data conds_lim;
	set M160_Out.cond_cred_tune_feat;
	where member_id in (
		&interesting_mems.
		);
run;

/*This file comes from prod03_fit_recommender.py, and was exported to SAS for convenience*/
proc export
	data = M160_Out.explanation_sample
	outfile = "&M160_Out.explanation_sample.xlsx"
	dbms = excel replace;
run;

proc export
	data = predictions_lim
	outfile = "&M160_Out.explanation_sample.xlsx"
	dbms = excel replace;
run;
proc export
	data = conds_lim
	outfile = "&M160_Out.explanation_sample.xlsx"
	dbms = excel replace;
run;
