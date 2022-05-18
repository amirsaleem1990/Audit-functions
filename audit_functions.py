#!/home/amir/Audit/Virtual-env-audit/bin/python3
import pickle
import hashlib
from itertools import groupby
import pandas as pd
from IPython.display import display
from termcolor import colored

class Audit:
	"""
	This class is perform some genarice data auditing, and save results in a local file.
	"""
	def __init__(self, data_frame, checks_dict: dict, data_name: str, all_variables=True):

		self.all_variables = all_variables
		self.data_frame = data_frame
		self.checks_dict = checks_dict
		self.results_lst = []
		self.cols_finished = []
		self.data_name = data_name

		# We try to change all string columns to float.
		for col in self.data_frame.select_dtypes("O"):
			try:
				self.data_frame[col] = self.data_frame[col].astype(float)
				self.data_frame[col] = self.data_frame[col].astype(int)
			except:
				pass


		# We try to change all string columns to datetime
		self.data_frame[self.data_frame.select_dtypes("O").columns.to_list()] = self.data_frame.select_dtypes("O").apply(
			lambda col: pd.to_datetime(col, errors='ignore')  if col.dtypes == object  else col
			)

		assert (
			(
				(self.all_variables) & (not len(self.checks_dict))
				) | (
				(not self.all_variables) & (len(self.checks_dict))
				)
			)

        # self.empty_string_ = np.vectorize(self.empty_string)
		# self.only_numbers_ = np.vectorize(self.only_numbers)
		# self.extra_space_ = np.vectorize(self.extra_space)
		# self.trailing_space_ = np.vectorize(self.trailing_space)
		# self.leading_space_ = np.vectorize(self.leading_space)
		# self.less_then_n_unique_chars_ = np.vectorize(self.less_then_n_unique_chars)
		# self.case_problem_ = np.vectorize(self.case_problem)
		# self.only_one_unique_value_in_entire_column_ = np.vectorize(self.only_one_unique_value_in_entire_column)
		# self.nan_ = np.vectorize(self.nan)

		self.string_funcs = [
							self.empty_string,
							self.only_numbers,
							self.extra_space,
							self.trailing_space,
							self.leading_space,
							self.lessazz_then_n_unique_chars,
							self.case_problem,
							self.only_one_unique_value_in_entire_column,
							self.nan,
							self.contains_punctuations,
							self.only_punctuations 
		]

		self.all_funcs = [
			self.same_char_more_than_n_times_in_sequence,
			self.only_n_chars
		]

		self.for_specific_cols = [
			self.wrong_mobile_number
		]


		self.date_funcs = [
			self.date_after,
			self.date_before
		]



	def summary(self):
		"""
		We audited, Now it time to summarise.
		"""
		cols_left = self.data_frame.columns.difference(self.cols_finished).to_list()
		if cols_left:
			print(colored("\n\nNo function applied to this/these column/s", "red"))
			print(self.data_frame[cols_left].dtypes.to_string())
			print("\n\n")

		file_name = f"results_lst_{self.data_name}"
		pickle.dump(self.results_lst, open(file_name+".pkl", 'wb'))
		self.results_lst = [i for i in self.results_lst if not i is None]
		df = (
			pd
			.DataFrame(self.results_lst)
			.dropna(subset=["Count"])
			.where(lambda x: x.Count.gt(0))
			.dropna(how='all', axis=0)
			.reset_index(drop=True)
			.dropna(how='all', axis=1)
		)
		df.to_csv(file_name+".csv", index=False)
		print(f"\nResults saved as <{file_name}.pkl> and <{file_name}.csv>\n\n")
		display(df)

		print("\n\n")

		display(
			df
			.groupby("Check")
			.agg({"Check" : len, "Count" : sum})
			.rename(columns={"Check" : "Check_qty", "Count" : "Row_count"})
			.astype(int)
			.assign(Avg_count=lambda x: x.Row_count // x.Check_qty)
			.reset_index()
			.sort_values(["Row_count", "Check_qty"], ascending=False)
		)

	def main(self):
		"""
		It's our main method, We call all functions defined in this class from here.
		"""

		self.repeated_columns()

		if self.all_variables:
			for col in self.data_frame:
				if self.data_frame[col].dtype == "O":
					for func_name in self.string_funcs:
						func_name(series=self.data_frame[col])
						self.cols_finished.append(col)
				if self.data_frame[col].dtype in ["float64", "int64"]:
					# We can't run our audit on numeric data.
					# for func_name in self.numeric_funcs:
					#   self.results_lst.append(func_name(series=self.data_frame[col]))
						self.cols_finished.append(col)
				if self.data_frame[col].dtype == 'datetime64[ns]':
					for func_name in self.date_funcs:
						func_name(series=self.data_frame[col])
						self.cols_finished.append(col)

				for func_name in self.all_funcs:
					func_name(series=self.data_frame[col])
					self.cols_finished.append(col)
		else:
			for col, func_name in self.checks_dict.items():
				self.results_lst.append(func_name(series=self.data_frame[col]))
				self.cols_finished.append(col)


		self.summary()


	def empty_string(self, series):
		x = series.astype(str).str.strip().eq("")

		self.results_lst.append({
			"Column" : series.name,
			"Check" : "empty_string",
			"Count" : x.sum(),
			"Perc" : round(x.mean()*100, 2)
		})

	def only_numbers(self, series):
		x = series.astype(str).str.strip().str.split(".0", -1).str[0]
		self.results_lst.append({
			"Column" : series.name,
			"Check" : "only_numbers",
			"Count" : x.str.isnumeric().sum(),
			"Perc" : round(x.str.isnumeric().mean()*100, 2),
			"Sample" : x[x.str.isnumeric()].value_counts().drop(["nan", ''], errors="ignore").head().index.to_list()
		})

	def extra_space(self, series):
		x = series.astype(str).str.contains("  ")
		self.results_lst.append({
			"Column" : series.name,
			"Check" : "extra_space",
			"Count" : x.sum(),
			"Perc" : round(x.mean()*100, 2),
			"Sample" : series[x].value_counts().head().index.to_list()
		})

	def trailing_space(self, series):
		x = series.astype(str).str.endswith(" ")
		self.results_lst.append({
			"Column" : series.name,
			"Check" : "trailing_space",
			"Count" : x.sum(),
			"Perc" : round(x.mean()*100, 2),
			"Sample" : series[x].value_counts().head().index.to_list()
		})

	def leading_space(self, series):
		x = series.astype(str).str.startswith(" ")
		self.results_lst.append({
			"Column" : series.name,
			"Check" : "leading_space",
			"Count" : x.sum(),
			"Perc" : round(x.mean()*100, 2),
			"Sample" : series[x].value_counts().head().index.to_list()
		})

	def same_char_more_than_n_times_in_sequence(self, series, n=4):
		def internal_function(string: str, n: int):
			if string is None:
				return False
			for _, v in groupby(string):
				if len(list(v)) > n:
					return True
			return False
		series = series.dropna().astype(str)
		x = series.apply(internal_function, n=n)
		self.results_lst.append({
			"Column" : series.name,
			"Check" : "same_char_more_than_n_times_in_sequence",
			"Count" : x.sum(),
			"Perc" : round(x.mean()*100, 2),
			"Sample" : series[x].value_counts().head().index.to_list()
		})



	def less_then_n_unique_chars(self, series, n=5):
		series = series.dropna()
		x = series.astype(str).apply(lambda x: len(set(x)) < 5)
		self.results_lst.append({
			"Column" : series.name,
			"Check" : f"less_then_{n}_unique_chars",
			"Count" : x.sum(),
			"Perc" : round(x.mean()*100, 2),
			"Sample" : series[x].dropna().value_counts().drop(["nan", ''], errors='ignore').head().index.to_list()
		})


	def only_one_unique_value_in_entire_column(self, series):
		self.results_lst.append({
			"Column" : series.name,
			"Check" : "only_one_unique_value_in_entire_column",
			"Value" : series.is_unique
		})

	def nan(self, series):
		self.results_lst.append({
			"Column" : series.name,
			"Check" : "nan",
			"Count" : series.isna().sum(),
			"Perc" : round(series.isna().mean()*100, 2)
		})

	def case_problem(self, series):
		unique_vals = pd.Series(series.unique())
		x = (
			unique_vals
			.groupby(
				unique_vals
				.astype(str)
				.str
				.lower()
			)
			.size()
			.where(lambda x: x > 2)
			.dropna()
			.index
			.to_list()
		)
		if not x:
			self.results_lst.append({
				"Column" : series.name,
				'Check': 'case_problem',
				 'Count': 0,
				 'Perc': 0,
				 'Sample': []
			})

		x2 = series.isin(x)
		self.results_lst.append({
			"Column" : series.name,
			"Check" : "case_problem",
			"Count" : x2.sum(),
			"Perc" : round(x2.mean()*100, 2),
			"Sample" :     list(
			series[
				series
				.str
				.lower()
				.eq(
					x
					[0]
				)
			].unique()
			)
		})


	def wrong_mobile_number(self, series):
		series = series.dropna().str.replace(r'[^0-9]+', '')
		wrong_values = (
			(series.str.startswith("03") & (series.str.len() != 11)) |#    03323388625 #11
			(series.str.startswith("0092") & (series.str.len() != 14)) |# 00923323388625 #14
			(series.str.startswith("92") & (series.str.len() != 12)) |#   923323388625 #12
			(series.str.startswith("3") & (series.str.len() != 10))#     3323388625 #10 
			)
		self.results_lst.append({
			"Column" : series.name,
			'Check': 'wrong_mobile_number',
			'Count': wrong_values.sum(),
			'Perc': round(wrong_values.mean()*100, 2),
			'Sample': series[wrong_values].value_counts().head().index.to_list()
		})


	def repeated_columns(self):
		hash_ = self.data_frame.apply(lambda col: hashlib.sha256(col.to_csv().encode()).hexdigest())
		x = hash_.duplicated(keep=False)
		if x.sum():
			print("\n\nThese columns are IDENDICAL\n" + "\n".join(*hash_[x].index.to_list()))


	def contains_punctuations(self, series):
		series = series.dropna().str.strip()
		punctuations="~|`|!|@|#|\$|%|\^|\&|\*|\(|\)|\+|{|}|\\|\||\?|\.|<|>"
		x = series.str.contains(punctuations).eq(True)
		if x.sum():
			self.results_lst.append({
				"Column" : series.name,
				'Check': 'contains_punctuations',
				'Count': x.sum(),
				'Perc': round(x.mean()*100, 2),
				'Sample': series[x].value_counts().head().index.to_list()
			})

	def only_punctuations(self, series):
		series = series.where(lambda x: x.str.strip().ne("")).dropna()
		punctuation_list = ["~", "`", "!", "@", "#", "\$", "%", "\^", "\&", "\*", "\(", "\)", "\+", "{", "}", "\\", "\|", "\?", "\.", "<", ">"]
		x = series.apply(lambda string: all([i in punctuation_list for i in string]))
		if x.sum():
			self.results_lst.append({
				"Column" : series.name,
				'Check': 'only_punctuations',
				'Count': x.sum(),
				'Perc': round(x.mean() *100, 2),
				'Sample': series[x].value_counts().head().index.to_list()
			})

	def only_n_chars(self, series, n=4):
		series = series.dropna()
		x = series.astype(str).str.strip().str.len().eq(n)
		self.results_lst.append({
			"Column" : series.name,
			'Check': f'only_n_chars_({n})',
			'Count': x.sum(),
			'Perc': round(x.mean() *100, 2),
			'Sample': series[x].value_counts().head().index.to_list()
		})

	def date_after(self, series, date):
		series = series.dropna()
		x = series.gt(date)
		self.results_lst.append({
			"Column" : series.name,
			'Check': f'date_after_({date})',
			'Count': x.sum(),
			'Perc': round(x.mean() *100, 2),
			'Sample': series[x].value_counts().head().index.to_list()
		})

	def date_before(self, series, date):
		series = series.dropna()
		x = series.lt(date)
		self.results_lst.append({
			"Column" : series.name,
			'Check': f'date_before_({date})',
			'Count': x.sum(),
			'Perc': round(x.mean() *100, 2),
			'Sample': series[x].value_counts().head().index.to_list()
		})


def display_sample(data_frame):
	pd.options.display.max_colwidth = None
	pd.options.display.max_columns = None
	x = data_frame.apply(lambda x: x.dropna().value_counts().head().index.to_list())
	df = pd.DataFrame(x.to_list(), index=x.index.to_list()).T
	display(df)
	# df.to_html("results_lst.html")

customer_master = pd.read_pickle("/home/amir/github/LFD_projects_4/33-Frontier/customer_master_0.pkl") 
pd.options.display.max_columns = None

audit_obt = Audit(data_frame=customer_master, checks_dict={}, data_name="customer_master")
audit_obt.main()
