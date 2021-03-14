import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import chi2_contingency, shapiro, normaltest, probplot, mannwhitneyu
from IPython.display import display


class Eda:
    def perform_for_numerical_feature(self, source_train_df, source_test_df, feature, target_feature):
        """

        :param source_train_df:
        :type source_train_df: pd.DataFrame
        :param source_test_df:
        :type source_test_df: pd.DataFrame
        :param feature:
        :type feature: str
        :param target_feature:
        :type target_feature: str
        """
        train_df = source_train_df.copy()
        test_df = source_test_df.copy()

        display(train_df[[feature]].describe())

        display(train_df[[feature]].info())

        na_items = train_df[train_df[feature].isna()][feature]
        display('Total NA: {}, has coverage: {}%'.format(na_items.size, na_items.size / train_df[feature].size * 100))

        train_df[[feature]].hist(log=True, figsize=(10, 8))

        abs(train_df[[feature]].corrwith(train_df[target_feature]))

        eda_feature_with_target_feature_df = train_df[[feature, target_feature]]
        eda_feature_df = eda_feature_with_target_feature_df[feature]

        eda_feature_target_0_df = eda_feature_df[eda_feature_with_target_feature_df[target_feature] == 0]
        eda_feature_target_1_df = eda_feature_df[eda_feature_with_target_feature_df[target_feature] == 1]

        plt.figure(figsize=(10, 8))

        sns.kdeplot(eda_feature_target_0_df, shade=True, label='Repayment', color='r')
        sns.kdeplot(eda_feature_target_1_df, shade=True, label='Overdue', color='g')

        plt.xlabel(feature)
        plt.title('%s grouped by target variable' % feature)
        plt.show()

        normaltest(eda_feature_df[~eda_feature_df.isna()])

        plt.figure(figsize=(20, 8))

        ax1 = plt.subplot(121)
        ax1.set_xlabel(feature)
        ax1.set_ylabel('Count')
        ax1.set_title('%s distribution' % feature)
        eda_feature_df.hist()

        plt.subplot(122)
        probplot(eda_feature_df, dist='norm', plot=plt)

        plt.show()

        mannwhitneyu(eda_feature_target_0_df, eda_feature_target_1_df)

        plt.figure(figsize=(10, 8))

        sns.pointplot(x=target_feature, y=feature, data=eda_feature_with_target_feature_df, capsize=.1)

        plt.title('Confidence intervals (95 %) for {}'.format(feature))
        plt.show()

        plt.figure(figsize=(10, 5))

        sns.set(font_scale=0.8)

        sns.kdeplot(train_df[feature], shade=True, label='train', color='r')
        sns.kdeplot(test_df[feature], shade=True, label='test', color='g')

        print(mannwhitneyu(train_df[feature], test_df[feature]))
        plt.legend()
        plt.title(feature)
        plt.show()

        for col in train_df.select_dtypes(include=[object]).columns.tolist():
            plt.figure(figsize=(20, 8))

            ax = sns.pointplot(x=col, y=feature, data=train_df, capsize=.1, label='train', color='r')
            ax_x_labels = ax.get_xticklabels()
            ax.set_xticklabels(ax_x_labels, rotation=90)

            ax = sns.pointplot(x=col, y=feature, data=test_df, capsize=.1, label='test', color='g')
            ax_x_labels = ax.get_xticklabels()
            ax.set_xticklabels(ax_x_labels, rotation=90)

            plt.title(col)
            plt.show()

    def perform_for_categorical_feature(self, source_train_df, source_test_df, feature, target_feature):
        """

        :param source_train_df:
        :type source_train_df: pd.DataFrame
        :param source_test_df:
        :type source_test_df: pd.DataFrame
        :param feature:
        :type feature: str
        :param target_feature:
        :type target_feature: str
        """
        train_df = source_train_df.copy()
        test_df = source_test_df.copy()

        display(train_df[[feature]].describe())

        display(train_df[[feature]].info())

        na_items = train_df[train_df[feature].isna()][feature]
        display('Total NA: {}, has coverage: {}%'.format(na_items.size, na_items.size / train_df[feature].size * 100))

        plt.figure(figsize=(10, 8))

        ax = sns.countplot(x=feature, hue=target_feature, data=train_df)
        ax_x_labels = ax.get_xticklabels()
        ax.set_xticklabels(ax_x_labels, rotation=90)
        plt.title('%s grouped by target variable' % feature)
        plt.legend(title='Target', loc='upper right')
        plt.show()

        plt.figure(figsize=(10, 8))

        ax = sns.pointplot(x=feature, y=target_feature, data=train_df, capsize=.1)
        ax_x_labels = ax.get_xticklabels()
        ax.set_xticklabels(ax_x_labels, rotation=90)