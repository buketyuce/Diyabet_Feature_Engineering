###############################################
# İş Problemi
###############################################
#Özellikleri belirtildiğinde kişilerin diyabet hastası olup olmadıklarını tahmin  edebilecek bir makine öğrenmesi modeli geliştirilmesi istenmektedir.
#Modeli geliştirmeden önce gerekli olan veri analizi ve özellik mühendisliği adımlarını gerçekleştirmemiz gerekmekte.

###############################################
# Veri Seti Hikayesi
###############################################
#Veri seti ABD'deki Ulusal Diyabet-Sindirim-Böbrek Hastalıkları Enstitüleri'nde tutulan büyük veri setinin parçasıdır.
#ABD'deki Arizona Eyaleti'nin en büyük 5. şehri olan Phoenix şehrinde yaşayan 21 yaş ve üzerinde olan Pima Indian kadınları üzerinde yapılan diyabet araştırması için kullanılan verilerdir.
#Hedef değişken "outcome" olarak belirtilmiş olup; 1 diyabet test sonucunun pozitif oluşunu, 0 ise negatif oluşunu belirtmektedir.

###############################################
# Değişkenler
###############################################
#Pregnancies: Hamilelik sayısı
#Glucose: Oral glikoz tolerans testinde 2 saatlik plazma glikoz konsantrasyonu
#Blood Pressure: Kan Basıncı (Küçük tansiyon) (mm Hg)
#SkinThickness: Cilt Kalınlığı
#Insulin: 2 saatlik serum insülini (mu U/ml)
#DiabetesPedigreeFunction: Soydaki kişilere göre diyabet olma ihtimalini hesaplayan bir fonksiyon
#BMI: Vücut kitle endeksi
#Age: Yaş (yıl)
#Outcome: Hastalığa sahip (1) ya da değil (0)


import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import  StandardScaler
import warnings
warnings.simplefilter(action="ignore")

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', None)

df = pd.read_csv("datasets/diabetes.csv")
df.head()


#Keşifçi Veri Analizi

#Genel resme baktım.
def check_df(dataframe, head=5):
    # boyut bilgisi
    print("################### Shape ###################")
    print(dataframe.shape)
    # tip bilgisi
    print("################### Types ###################")
    print(dataframe.dtypes)
    #Baştan gözlemleyelim
    print("################### Head ###################")
    print(dataframe.head(head))
    #Sondan gözlemleyelim
    print("################### Tail ###################")
    print(dataframe.tail(head))
    #Veri setinde herhangi bir eksik değer var mı bakalım
    print("################### NA ###################")
    print(dataframe.isnull().sum())
    #Sayısal değişkenlerin dağılımına bakalım
    print("################### Quantiles ###################")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)


#Numerik ve kategorik değişkenleri yakaladım.
def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.
    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri
    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi
    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))
    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı
    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

grab_col_names(df)

num_cols =  ['Pregnancies',
  'Glucose',
  'BloodPressure',
  'SkinThickness',
  'Insulin',
  'BMI',
  'DiabetesPedigreeFunction',
  'Age']

#Numerik ve kategorik değişkenlerin analizini yaptım.

#Kategorik
def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100* dataframe[col_name].value_counts()/len(dataframe)}))
    print("#######################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()


cat_summary(df, "Outcome")

#Nümerik
def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()

for col in num_cols:
    num_summary(df, col, plot=True)

#Hedef değişken analizi yaptım.

def target_summary_with_num(dataframe, target, numarical_col):
    print(dataframe.groupby(target).agg({numarical_col: "mean"}), end="\n\n\n")

for col in num_cols:
    target_summary_with_num(df, "Outcome", col)

#Korelasyon analizi yaptım.
df.corr()
f, ax = plt.subplots(figsize=[18,13])
sns.heatmap(df.corr(), annot=True, fmt= ".2f", ax=ax, cmap= "magma")
ax.set_title("Correlation Matrix", fontsize=20)
plt.show()

###############################################
#BASE MODEL KURULUMU
###############################################
y = df["Outcome"]
x = df.drop("Outcome" ,axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=17)

rf_model = RandomForestClassifier(random_state=46).fit(x_train, y_train)
y_pred = rf_model.predict(x_test)

print(f"Accuracy: {round(accuracy_score(y_pred, y_test),2)}")
print(f"Recall: {round(recall_score(y_pred, y_test), 3)}")
print(f"Precision: {round(precision_score(y_pred, y_test), 2)}")
print(f"F1: {round(f1_score(y_pred, y_test), 2)}")
print(f"Auc: {round(roc_auc_score(y_pred, y_test), 2)}")

#Accuracy: 0.77
#Recall: 0.706
#Precision: 0.59
#F1: 0.64
#Auc: 0.75

def plot_importance(model, features, num=len(x), save=False):
    feature_imp = pd.DataFrame({"Values": model.feature_importances_, "Feature": features.columns})
    plt.figure(figsize=(10,10))
    sns.set(font_scale=1)
    sns.barplot(x="Values", y="Feature", data=feature_imp.sort_values(by="Values",
                                                                      ascending=False)[0:num])
    plt.title("Features")
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig("importances.png")

plot_importance(rf_model, x)


###############################################
#Eksik değer analizi yaptım.
###############################################
#Bir insanda Pregnancies ve Outcome dışındaki değişken değerleri 0 olamayacağı bilinmektedir.
#Bundan dolayı bu değerlerle ilgili aksiyon kararı alınmalıdır. 0 olan değerlere NaN atanabilir.

zero_columns = [col for col in df.columns if (df[col].min() == 0 and col not in ["Pregnancies", "Outcome"])]
zero_columns

#Gözlem birimlerinde 0 olan değişkenlerin her birisine gidip 0 içeren gözlem değerlerini NaN ile değiştirdim.
for col in zero_columns:
    df[col] = np.where(df[col] == 0, np.nan, df[col])

#Eksik gmzlem analizi
df.isnull().sum()

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio,2)], axis=1, keys=["n_miss", "ratio"])
    print(missing_df, end="\n")
    if na_name:
        return na_columns

na_columns = missing_values_table(df, na_name=True)

#Eksik Değerlerin Bağımlı Değişken ile ilişkisinin İncelenmesi

def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()
    for col in na_columns:
        temp_df[col + "_NA_FLAG"] = np.where(temp_df[col].isnull(), 1, 0)
    na_flags= temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns
    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")

missing_vs_target(df, "Outcome", na_columns)

#Eksik Değerlerin Doldurulması

for col in zero_columns:
    df.loc[df[col].isnull(), col] = df[col].median()

df.isnull().sum()

###############################################
#Aykırı gözlem analizi
###############################################

def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit



def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


def replace_with_thresholds(dataframe, variable, q1=0.05, q3=0.95):
    low_limit, up_limit = outlier_thresholds(dataframe, variable, q1=0.05, q3=0.95)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

#AYKIRI DEĞER ANALİZİ VE BASKLAMA İŞLEMİ

for col in df.columns:
    print(col, check_outlier(df, col))
    if check_outlier(df, col):
        replace_with_thresholds(df,col)

for col in df.columns:
    print(col, check_outlier(df,col))

################################################
#ÖZELLİK ÇIKARIMI
###############################################
#Yaş değişkenini katagorilere ayırıp yeni yaş değişkeni oluşturulması
df.loc[(df["Age"] >= 21) & (df["Age"] < 50), "NEW_AGE_CAT"] = "mature"
df.loc[(df["Age"] >= 50), "NEW_AGE_CAT"] = "senior"

#BMI 18,5 asağısı underweight, 10.5 ile 24.9 arası normal, 24.9 ile 29.9 arası Overweight ve 30 üstü obese
df["NEW_BMI"] = pd.cut(x=df["BMI"], bins=[0, 10.5, 24.9, 29.9, 100], labels=["Underweight", "Healthy", "Overweight", "Obese"])

#Glukoz değerlerini kategorik değişkene çevirme
df["NEW_GLUCOSE"] = pd.cut(x=df["Glucose"], bins=[0,140,200,300], labels=["Normal", "Prediabetes", "Diabetes"])

#İnsülin değeri ile kategorik değişken türetme
def set_insulin(dataframe, col_name= "Insulin"):
    if 16 <= dataframe[col_name] <= 166:
        return "Normal"
    else:
        return "Anormal"

df["NEW_INSULIN_SCORE"] = df.apply(set_insulin, axis=1)

df["NEW_GLUCOSE*INSULIN"] = df["Glucose"] * df["Insulin"]

#kolonların isimlerini büyüttüm
df.columns = [col.upper() for col in df.columns]

df.head()

################################################
#ENCODING
################################################

# Değişkenlerin tiplerine göre ayrılması işlemi
cat_cols, num_cols, cat_but_car = grab_col_names(df)

#LABEL ENCODİNG

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = [col for col in df.columns if df[col].dtypes == "O" and df[col].nunique() == 2]

for col in binary_cols:
    df = label_encoder(df, col)

df.head()

#One-Hot Encoding İşlemi
#cat_cols listesinin güncelleme işlemi

cat_cols = [col for col in cat_cols if col not in binary_cols and col not in ["OUTCOME"]]

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

df = one_hot_encoder(df, cat_cols, drop_first=True)

df.head()

################################################
#STANDARTLAŞTIRMA
################################################

num_cols

scaler = StandardScaler()
df[num_cols] =scaler.fit_transform(df[num_cols])

df.head()

df.shape

################################################
#MODELLEME
################################################
y = df["OUTCOME"]
X = df.drop("OUTCOME", axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

print(f"Accuracy: {round(accuracy_score(y_pred, y_test),2)}")
print(f"Recall: {round(recall_score(y_pred, y_test), 3)}")
print(f"Precision: {round(precision_score(y_pred, y_test), 2)}")
print(f"F1: {round(f1_score(y_pred, y_test), 2)}")
print(f"Auc: {round(roc_auc_score(y_pred, y_test), 2)}")

#Accuracy: 0.78
#Recall: 0.703
#Precision: 0.64
#F1: 0.67
#Auc: 0.76

################################################
#FEATURE İMPORTANCE
################################################

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({"Value": model.feature_importances_, "Feature": features.columns})
    print(feature_imp.sort_values("Value", ascending=False))
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data= feature_imp.sort_values(by="Value",
                                                                      ascending=False)[0:num])
    plt.title("Features")
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig("importances.png")

plot_importance(rf_model, X)