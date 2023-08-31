#!/usr/bin/env python
# coding: utf-8

# # Modüllerin içe aktarılması ve verilerin yüklenmesi

# In[1]:


import numpy as mp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn

veriler=pd.read_csv("C:\\Users\\Dell\\Desktop\\datathon2023\\train.csv")
testVerileri=pd.read_csv("C:\\Users\\Dell\\Desktop\\datathon2023\\test_x.csv")


# # Verilere genel bir bakış

# In[2]:


print(veriler.isnull().sum())
print("------------------------")
veriler.info()
# verilerde her değişkende 5460 değer bulunuyor ve hiç kayıp görünmüyor.


# In[3]:


print(testVerileri.isnull().sum())
print("------------------------")
testVerileri.info()
# verilerde her değişkende 2340 değer bulunuyor ve hiç kayıp görünmüyor.


# In[4]:


sbn.displot(data=veriler,x="Yıllık Ortalama Gelir", kde=True)
#veri normal dağılım göstermemekte ve aykırı değerler içermektedir.


# In[5]:


sbn.displot(data=veriler,x="Yıllık Ortalama Satın Alım Miktarı", kde=True)
#veri normal dağılım göstermemektedir.


# In[6]:


sbn.displot(data=veriler,x="Yıllık Ortalama Sipariş Verilen Ürün Adedi", kde=True)
#veri normal dağılım göstermemektedir.


# In[7]:


sbn.displot(data=veriler,x="Yıllık Ortalama Sepete Atılan Ürün Adedi", kde=True)
#veri normal dağılım göstermemektedir.


# In[8]:


plt.boxplot(veriler["Yıllık Ortalama Gelir"])
# grafikte aykırı değer görülmektedir. grafiye göre eğik ve negatif bir durum söz konusudur. 


# In[9]:


plt.boxplot(veriler["Yıllık Ortalama Satın Alım Miktarı"])
#eğik ve negatif bir durum söz konusudur. veriler üst limite daha yakındır.


# In[10]:


plt.boxplot(veriler["Yıllık Ortalama Sipariş Verilen Ürün Adedi"])
#eğik ve negatif bir durum söz konusudur. veriler üst limite daha yakındır.


# In[11]:


plt.boxplot(veriler["Yıllık Ortalama Sepete Atılan Ürün Adedi"])
# grafikte aykırı değer görülmektedir. #eğik ve pozitif bir durum söz konusudur. veriler alt limite daha yakındır.


# In[12]:


sbn.countplot(data=veriler,x="Cinsiyet")
#erkek sayısının kadın sayısına göre biraz daha fazladır.


# In[13]:


sbn.countplot(data=veriler,x="Yaş Grubu")
#yaş dağılımında genç kategoriden yaşlı kategoriye doğru bir azalış eğilimi göstermektedir.


# In[14]:


sbn.countplot(data=veriler,x="Medeni Durum")
#evli sayısı fazla olmasına rağmen dengeli bir dağılıma sahiptir.


# In[15]:


sbn.countplot(data=veriler,x="İstihdam Durumu")
plt.xticks(rotation=-45)


# In[16]:


sbn.countplot(data=veriler,x="Yaşadığı Şehir")


# In[17]:


sbn.countplot(data=veriler,x="En Çok İlgilendiği Ürün Grubu")
plt.xticks(rotation=-45)


# In[18]:


sbn.countplot(data=veriler,x="Eğitime Devam Etme Durumu")


# In[19]:


sbn.countplot(data=veriler,x="Öbek İsmi"),
#endüşük kayıt sayısı obek 2 olarak görünmekte diğerleri dengeli bir dağılıma sahiptir.


# In[20]:


veriler.describe()


# In[21]:


veriler.describe(include=['O'])


# In[22]:


testVerileri.describe()


# In[23]:


testVerileri.describe(include=['O'])


# In[24]:


veriler.rename(columns={"Öbek İsmi":"ObekIsmi"},inplace=True)
obekListesi=["obek_1","obek_2","obek_3","obek_4","obek_5","obek_6","obek_7","obek_8"]


# In[25]:


# sayısal verilerin okunabilirliğini arttırmak için her bir sutunu bir tablo haline dönüştürüyoruz.
def olustur(sutun):    
    tablo1=veriler[sutun]
    tablo1=pd.concat([tablo1,veriler["ObekIsmi"]],axis=1)    
    tablo2=pd.DataFrame(index=["Count","Sum","Mean","Std","Min","Max","Yüzde"],columns=["obek_1","obek_2","obek_3",
                                                                                             "obek_4","obek_5","obek_6",
                                                                                             "obek_7","obek_8"])         
    for i in obekListesi:
        g=0
        liste=[]
        while g<len(tablo1):
            if tablo1["ObekIsmi"][g]==i:
                liste.append(tablo1[sutun][g])
            g=g+1
        liste=mp.array(liste)
        toplam=liste.sum()
        say=len(liste)
        ort=liste.mean()
        stds=liste.std()
        mind=liste.min()
        mak=liste.max()
        tablo2[i]["Sum"]=int(toplam)
        tablo2[i]["Mean"]=int(ort)
        tablo2[i]["Std"]=int(stds)
        tablo2[i]["Min"]=int(mind)
        tablo2[i]["Max"]=int(mak)
        tablo2[i]["Count"]=say
        tablo2[i]["Yüzde"]=say/len(tablo1)
    return tablo2


# In[26]:


# kategorik verilerin okunabilirliğini arttırmak için her bir sutunu bir tablo haline dönüştürüyoruz.
sutunListesi=["Cinsiyet","Yaş Grubu","Medeni Durum","Eğitim Düzeyi","İstihdam Durumu","Yaşadığı Şehir",
              "En Çok İlgilendiği Ürün Grubu","Eğitime Devam Etme Durumu"]

cinsiyet=pd.DataFrame()
yasGrubu=pd.DataFrame()
medeniDurum=pd.DataFrame()
egitimDuzeyi=pd.DataFrame()
istihdamDurumu=pd.DataFrame()
yasadigiSehir=pd.DataFrame()
urunGrubu=pd.DataFrame()
mezunDurumu=pd.DataFrame()

for i in obekListesi:
    tablo=veriler[(veriler.ObekIsmi==i)]    
    for k in sutunListesi:
        if k=="Cinsiyet":
            tablo2=pd.DataFrame(tablo[k].value_counts())
            tablo2.rename(columns={"count":i},inplace=True)
            cinsiyet=pd.concat([cinsiyet,tablo2],axis=1)
        if k=="Yaş Grubu":
            tablo2=pd.DataFrame(tablo[k].value_counts())
            tablo2.rename(columns={"count":i},inplace=True)
            yasGrubu=pd.concat([yasGrubu,tablo2],axis=1)
        if k=="Medeni Durum":
            tablo2=pd.DataFrame(tablo[k].value_counts())
            tablo2.rename(columns={"count":i},inplace=True)
            medeniDurum=pd.concat([medeniDurum,tablo2],axis=1)
        if k=="Eğitim Düzeyi":
            tablo2=pd.DataFrame(tablo[k].value_counts())
            tablo2.rename(columns={"count":i},inplace=True)
            egitimDuzeyi=pd.concat([egitimDuzeyi,tablo2],axis=1)
        if k=="İstihdam Durumu":
            tablo2=pd.DataFrame(tablo[k].value_counts())
            tablo2.rename(columns={"count":i},inplace=True)
            istihdamDurumu=pd.concat([istihdamDurumu,tablo2],axis=1)
        if k=="Yaşadığı Şehir":
            tablo2=pd.DataFrame(tablo[k].value_counts())
            tablo2.rename(columns={"count":i},inplace=True)
            yasadigiSehir=pd.concat([yasadigiSehir,tablo2],axis=1)
        if k=="En Çok İlgilendiği Ürün Grubu":
            tablo2=pd.DataFrame(tablo[k].value_counts())
            tablo2.rename(columns={"count":i},inplace=True)
            urunGrubu=pd.concat([urunGrubu,tablo2],axis=1)
        if k=="Eğitime Devam Etme Durumu":
            tablo2=pd.DataFrame(tablo[k].value_counts())
            tablo2.rename(columns={"count":i},inplace=True)
            mezunDurumu=pd.concat([mezunDurumu,tablo2],axis=1)     


# In[27]:


#öbeklere göre cinsiyet dağılımı
cinsiyet


# In[28]:


#öbeklere göre yaş dağılımı
yasGrubu


# In[29]:


#öbeklere göre medeni durum dağılımı
medeniDurum


# In[30]:


#öbeklere göre eğitim düzeyi dağılımı
egitimDuzeyi


# In[31]:


#öbeklere göre istihdam dağılımı
istihdamDurumu


# In[32]:


##öbeklere göre yaşadığı şehir dağılımı
yasadigiSehir


# In[33]:


#öbeklere göre ürün grubu dağılımı
urunGrubu


# In[34]:


#öbeklere göre mezuniyet dağılımı
mezunDurumu


# In[35]:


#öbeklere göre gelir dağılımı
gelirTablosu=olustur("Yıllık Ortalama Gelir")
gelirTablosu.index.names=["Yıllık Ort. Gelir Tablosu"]
gelirTablosu


# In[36]:


#öbeklere göre sipariş verilen ürün miktar dağılımı
urunAdediTablosu=olustur("Yıllık Ortalama Sipariş Verilen Ürün Adedi")
urunAdediTablosu.index.names=["Y. Ort. Sipariş Ver. Ürün Adet Tablosu"]
urunAdediTablosu


# In[37]:


#öbeklere göre satım alım miktar dağılımı
satinAlmaTablosu=olustur("Yıllık Ortalama Satın Alım Miktarı")
satinAlmaTablosu.index.names=["Yıllık Ort. Satın Alım Miktar Tablosu"]
satinAlmaTablosu


# In[38]:


#öbeklere göre sepete atılan ürün miktar dağılımı
sepeteAtmaTablosu=olustur("Yıllık Ortalama Sepete Atılan Ürün Adedi")
sepeteAtmaTablosu.index.names=["Yıllık Ort. Sepete Atılan Ürün Miktar Tablosu"]
sepeteAtmaTablosu


# In[39]:


# Makine öğrenmesinde ezber olmaması için id bilgilerini çıkartarak verileri yeni bir değişkene atıyoruz.
x=veriler.iloc[:,1:]
xt=testVerileri.iloc[:,1:]


# # Encoding İşlemi

# In[40]:


# Öbek ismi sutunundaki metisel veriyi sayısal veriye dönüştürüyoruz.
i=0
while i<len(x):
    if x["ObekIsmi"][i]=="obek_1":
        x["ObekIsmi"][i]=0
    if x["ObekIsmi"][i]=="obek_2":
        x["ObekIsmi"][i]=1
    if x["ObekIsmi"][i]=="obek_3":
        x["ObekIsmi"][i]=2
    if x["ObekIsmi"][i]=="obek_4":
        x["ObekIsmi"][i]=3
    if x["ObekIsmi"][i]=="obek_5":
        x["ObekIsmi"][i]=4
    if x["ObekIsmi"][i]=="obek_6":
        x["ObekIsmi"][i]=5
    if x["ObekIsmi"][i]=="obek_7":
        x["ObekIsmi"][i]=6
    if x["ObekIsmi"][i]=="obek_8":
        x["ObekIsmi"][i]=7
    i=i+1    


# In[41]:


# x tablosunu genel bir bakış
x.info()


# In[42]:


# öbek ismi sutununu eğitimde hata vermemesi için object veri türünden integer veri türüne dönüştürüyoruz.
x["ObekIsmi"]=pd.to_numeric(x["ObekIsmi"], downcast='integer')


# In[43]:


# verilerin label encoding ve one hot encoding işlemlerinin yapılması
x=pd.get_dummies(data=x,columns=["Cinsiyet","Yaş Grubu","Medeni Durum","Eğitim Düzeyi",
                               "İstihdam Durumu","Yaşadığı Şehir",
                               "En Çok İlgilendiği Ürün Grubu","Eğitime Devam Etme Durumu"
                               ],dtype=float)
xt=pd.get_dummies(data=xt,columns=["Cinsiyet","Yaş Grubu","Medeni Durum","Eğitim Düzeyi",
                               "İstihdam Durumu","Yaşadığı Şehir",
                               "En Çok İlgilendiği Ürün Grubu","Eğitime Devam Etme Durumu"
                               ],dtype=float)


# In[44]:


#verilerin encoding işleminden sonra sutunların veri tipi ve eğitim ve test veri setindeki sutun sıralamasının
# kontrolunun yapılması
x.info()


# In[45]:


xt.info()


# # Verilerin Eğitim İçin Bölünmesi ve Ölçüklendirilmesi

# In[46]:


y=x.iloc[:,3:4].values
x.drop("ObekIsmi",axis=1,inplace=True)
x=x.values
xt=xt.values


# In[47]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# In[48]:


from sklearn.preprocessing import StandardScaler
#from sklearn.preprocessing import PowerTransformer
#from sklearn.preprocessing import RobustScaler
#from sklearn.preprocessing import MinMaxScaler
sc=StandardScaler()
x_train=sc.fit_transform(X_train)
x_test=sc.transform(X_test)
xt_test=sc.transform(xt)


# In[49]:


#LDA
# Veri boyutlarını indirgemek ve sınıflar arasındaki mesafeyi maksimize etmek için tercih edildi. Veri normal dağılım
# göstermediği için PCA tercih edilmedi.
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda=LDA(n_components=6)
x_train_lda=lda.fit_transform(x_train,Y_train)
x_test_lda=lda.transform(x_test)
predict_test_lda=lda.transform(xt_test)


# In[50]:


#Başarı ölçüm mekriklerinin içe aktarılması
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


# # Makine Öğrenmesi Algoritmalarının Oluşturulması ve Eğitilmesi

# In[51]:


#Logistic Regression

rState=0
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=rState)
classifier.fit(x_train, Y_train)

y_pred_lr=classifier.predict(x_test)
cm_lr=confusion_matrix(Y_test, y_pred_lr)
print(cm_lr)
acc_lr=accuracy_score(Y_test, y_pred_lr)
print(acc_lr)

print("------LDA------")
classifier_lda=LogisticRegression(random_state=rState)
classifier_lda.fit(x_train_lda, Y_train)

y_pred_lr_lda=classifier_lda.predict(x_test_lda)
cm_lr_lda=confusion_matrix(Y_test,y_pred_lr_lda)
print(cm_lr_lda)
acc_lr_lda=accuracy_score(Y_test, y_pred_lr_lda)
print(acc_lr_lda)


# In[52]:


#KNN Classifier

n=30
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=n,metric="minkowski")
knn.fit(x_train, Y_train)

y_pred_knn=knn.predict(x_test)
cm_knn=confusion_matrix(Y_test, y_pred_knn)
print(cm_knn)
acc_knn=accuracy_score(Y_test, y_pred_knn)
print(acc_knn)

print("------LDA------")
knn_lda=KNeighborsClassifier(n_neighbors=n,metric="minkowski")
knn_lda.fit(x_train_lda, Y_train)

y_pred_knn_lda=knn_lda.predict(x_test_lda)
cm_knn_lda=confusion_matrix(Y_test,y_pred_knn_lda)
print(cm_knn_lda)
acc_knn_lda=accuracy_score(Y_test, y_pred_knn_lda)
print(acc_knn_lda)


# In[53]:


#SVM

kernel="linear"
from sklearn.svm import SVC
svc=SVC(kernel=kernel)
svc.fit(x_train,Y_train)

y_pred_svc=svc.predict(x_test)
cm_svc=confusion_matrix(Y_test,y_pred_svc)
print(cm_svc)
acc_svc=accuracy_score(Y_test,y_pred_svc)
print(acc_svc)

print("------LDA------")
svc_lda=SVC(kernel=kernel)
svc_lda.fit(x_train_lda, Y_train)

y_pred_svc_lda=svc_lda.predict(x_test_lda) 
cm_svc_lda=confusion_matrix(Y_test,y_pred_svc_lda)
print(cm_svc_lda)
acc_svc_lda=accuracy_score(Y_test, y_pred_svc_lda)
print(acc_svc_lda)


# In[54]:


#Naive Bayes

from sklearn.naive_bayes import GaussianNB
gnb=GaussianNB()
gnb.fit(x_train,Y_train)

y_pred_nb=gnb.predict(x_test)
cm_nb=confusion_matrix(Y_test,y_pred_nb)
print(cm_nb)
acc_nb=accuracy_score(Y_test,y_pred_nb)
print(acc_nb)

print("------LDA------")
gnb_lda=GaussianNB()
gnb_lda.fit(x_train_lda, Y_train)

y_pred_gnb_lda=gnb_lda.predict(x_test_lda) 
cm_gnb_lda=confusion_matrix(Y_test,y_pred_gnb_lda)
print(cm_gnb_lda)
acc_gnb_lda=accuracy_score(Y_test, y_pred_gnb_lda)
print(acc_gnb_lda)


# In[55]:


#Decision Tree

n=90
crit="entropy"
rState=5
from sklearn.tree import DecisionTreeClassifier
dtc=DecisionTreeClassifier(criterion=crit,random_state=rState)
dtc.fit(x_train,Y_train)

y_pred_dtc=dtc.predict(x_test)
cm_dtc=confusion_matrix(Y_test,y_pred_dtc)
print(cm_dtc)
acc_dtc=accuracy_score(Y_test,y_pred_dtc)
print(acc_dtc)

print("------LDA------")
dtc_lda=DecisionTreeClassifier(criterion=crit,random_state=rState)
dtc_lda.fit(x_train_lda, Y_train)

y_pred_dtc_lda=dtc_lda.predict(x_test_lda) 
cm_dtc_lda=confusion_matrix(Y_test,y_pred_dtc_lda)
print(cm_dtc_lda)
acc_dtc_lda=accuracy_score(Y_test, y_pred_dtc_lda)
print(acc_dtc_lda)


# In[56]:


#Random Forest

n=72
crit="entropy"
rState=16
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(n_estimators=n,criterion=crit,random_state=rState)
rfc.fit(x_train,Y_train)

y_pred_rfc=rfc.predict(x_test)
cm_rfc=confusion_matrix(Y_test,y_pred_rfc)
print(cm_rfc)
acc_rfc=accuracy_score(Y_test,y_pred_rfc)
print(acc_rfc)

print("------LDA------")
rfc_lda=RandomForestClassifier(n_estimators=n,criterion=crit,random_state=rState)
rfc_lda.fit(x_train_lda, Y_train)

y_pred_rfc_lda=dtc_lda.predict(x_test_lda) 
cm_rfc_lda=confusion_matrix(Y_test,y_pred_rfc_lda)
print(cm_rfc_lda)
acc_rfc_lda=accuracy_score(Y_test, y_pred_rfc_lda)
print(acc_rfc_lda)


# In[57]:


#XGboost

n=100
maxDept=10
lRate=0.01
rState=5
from xgboost import XGBClassifier
xgb=XGBClassifier(n_estimators=n, max_depth=maxDept, learning_rate=lRate,  random_state=rState)
xgb.fit(x_train,Y_train)

y_pred_xgb=xgb.predict(x_test)
cm_xgb=confusion_matrix(Y_test,y_pred_xgb)
print(cm_xgb)
acc_xgb=accuracy_score(Y_test,y_pred_xgb)
print(acc_xgb)

print("------LDA------")
xgb_lda=XGBClassifier(n_estimators=n, max_depth=maxDept, learning_rate=lRate,  random_state=rState)
xgb_lda.fit(x_train_lda, Y_train)

y_pred_xgb_lda=xgb_lda.predict(x_test_lda) 
cm_xgb_lda=confusion_matrix(Y_test,y_pred_xgb_lda)
print(cm_xgb_lda)
acc_xgb_lda=accuracy_score(Y_test, y_pred_xgb_lda)
print(acc_xgb_lda)


# In[58]:


#XGboost RFClassifier

n=95
maxDept=7
lRate=0.001
#obj='binary:logistic'
r_s_xgbrfc=7
from xgboost import XGBRFClassifier
xgbrfc=XGBRFClassifier(n_estimators=n, max_depth=maxDept, learning_rate=lRate,  random_state=r_s_xgbrfc)
xgbrfc.fit(x_train,Y_train)

y_pred_xgbrfc=xgbrfc.predict(x_test)
cm_xgbrfc=confusion_matrix(Y_test,y_pred_xgbrfc)
print(cm_xgb)
acc_xgbrfc=accuracy_score(Y_test,y_pred_xgbrfc)
print(acc_xgbrfc)

print("------LDA------")
xgbrfc_lda=XGBRFClassifier(n_estimators=n, max_depth=maxDept, learning_rate=lRate,  random_state=r_s_xgbrfc)
xgbrfc_lda.fit(x_train_lda, Y_train)

y_pred_xgbrfc_lda=xgbrfc_lda.predict(x_test_lda) 
cm_xgbrfc_lda=confusion_matrix(Y_test,y_pred_xgbrfc_lda)
print(cm_xgbrfc_lda)
acc_xgbrfc_lda=accuracy_score(Y_test, y_pred_xgbrfc_lda)
print(acc_xgbrfc_lda)


# In[59]:


#Stochastic Gradient Descent

from sklearn.linear_model import SGDClassifier
sgdc=SGDClassifier(max_iter=45,loss="hinge")
sgdc.fit(x_train,Y_train)

y_pred_sgdc=sgdc.predict(x_test)
cm_sgdc=confusion_matrix(Y_test,y_pred_sgdc)
print(cm_sgdc)
acc_sgdc=accuracy_score(Y_test,y_pred_sgdc)
print(acc_sgdc)

print("------LDA------")
sgdc_lda=sgdc=SGDClassifier(max_iter=21,loss="hinge",penalty="l1")
sgdc_lda.fit(x_train_lda, Y_train)

y_pred_sgdc_lda=sgdc_lda.predict(x_test_lda) 
cm_sgdc_lda=confusion_matrix(Y_test,y_pred_sgdc_lda)
print(cm_sgdc_lda)
acc_sgdc_lda=accuracy_score(Y_test, y_pred_sgdc_lda)
print(acc_sgdc_lda)


# In[60]:


#VotingClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBRFClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score

clf1=LogisticRegression(random_state=0)
clf2=RandomForestClassifier(n_estimators=78,criterion="entropy",random_state=14)
clf3=SVC(kernel="linear")
clf4=XGBRFClassifier(n_estimators=72, max_depth=9, learning_rate=0.01,  random_state=42)
clf5=XGBClassifier(n_estimators=100, max_depth=10, learning_rate=0.01,  random_state=5)

vclf=VotingClassifier(estimators=[("lr",clf1),("rfc",clf2),("svc",clf3),("xgbrfc",clf4),("xgbc",clf5)],voting="hard")

vclf.fit(x_train,Y_train)
y_pred_vclf=vclf.predict(x_test)
print(f"Vating Classifier Accuracy Score: {accuracy_score(Y_test, y_pred_vclf)}")


# In[61]:


#Sayısal veriye dönüştürdüğümüz Öbek İsmi sutununu tahmin dosyasını oluşturmak üzere tekrar metinsel ifadeye dönüştürüyoruz.
def donustur(liste):
    kolon=testVerileri.iloc[:,0:1]
    kolon.rename(columns={"index":"id"},inplace=True)
    i=0
    liste=list(liste)
    while i<len(liste):
        if liste[i]==0:
            liste[i]="obek_1"
        if liste[i]==1:
            liste[i]="obek_2"
        if liste[i]==2:
            liste[i]="obek_3"
        if liste[i]==3:
            liste[i]="obek_4"
        if liste[i]==4:
            liste[i]="obek_5"
        if liste[i]==5:
            liste[i]="obek_6"
        if liste[i]==6:
            liste[i]="obek_7"
        if liste[i]==7:
            liste[i]="obek_8"
        i=i+1
    liste=pd.DataFrame(data=liste,columns=["Öbek İsmi"])
    kolon=pd.concat([kolon,liste],axis=1)
    return kolon


# # Tahminler
# Logistic Regression Tahmin

# Parametreler

# test size oranı -> 0.2
# split random state -> 0
# Scaler -> Standart Scaler
# Lda -> Kullanıldı, n sayısı=7
# Random state -> 0
# Accuracy değeri -> 0,9478

predictList_lr_lda=classifier_lda.predict(predict_test_lda)
print(predictList_lr_lda)
sonuc=donustur(predictList_lr_lda)
print(sonuc)
sonuc.to_csv("C:\\Users\\Dell\\Desktop\\Lrc_lda predict list.csv",index=False)# Knn tahmin

# Parametreler

# test size oranı -> 0.2
# split random state -> 0
# Scaler -> Standart Scaler
# Lda -> Kullanıldı, n sayısı=7
# Knn n sayısı -> 30
# metrik -> minkowski
# Accuracy değeri -> 0,9551

predictList_knn_lda=knn_lda.predict(predict_test_lda)
print(predictList_knn_lda)
sonuc=donustur(predictList_knn_lda)
print(sonuc)
sonuc.to_csv("C:\\Users\\Dell\\Desktop\\Knn_lda predict list.csv",index=False)# Xgboost tahmin

# Parametreler

# test size oranı -> 0.2
# split random state -> 0
# Scaler -> Standart Scaler
# Lda -> Kullanılmadı
# Xgboost n sayısı -> 60
# Max dept -> 7
# Learnin rate -> 0.01
# Random state -> 5
# Accuracy değeri -> 0,9505

predictList_xgb=xgb.predict(xt_test)
print(predictList_xgb)
sonuc=donustur(predictList_xgb)
print(sonuc)
sonuc.to_csv("C:\\Users\\Dell\\Desktop\\XGBoost predict list.csv",index=False)# SVM tahmin

# test size oranı -> 0.2
# split random state -> 0
# Scaler -> Min Max Scaler
# Lda -> Kullanılmadı
# Kernel -> Linear
# Accuracy değeri -> 0,9523

predictList_svc=svc.predict(xt_test)
print(predictList_svc)
sonuc=donustur(predictList_svc)
print(sonuc)
sonuc.to_csv("C:\\Users\\Dell\\Desktop\\SVM predict list.csv",index=False)# Random Forest tahmin

# test size oranı -> 0.2
# split random state -> 0
# Scaler -> Min Max Scaler
# Lda -> Kullanılmadı
# Criterion -> Entropy
# RF n sayısı -> 78
# Random state -> 14
# Accuracy değeri -> 0,9551

predictList_rfc=rfc.predict(xt_test)
print(predictList_rfc)
sonuc=donustur(predictList_rfc)
print(sonuc)
sonuc.to_csv("C:\\Users\\Dell\\Desktop\\RFC predict list.csv",index=False)# XGBRFC tahmin

# Parametreler

# test size oranı -> 0.2
# split random state -> 0
# Scaler -> Min Max Scaler
# Lda -> Kullanıldı, n=6
# Xgbrfc n sayısı -> 72
# Max dept -> 9
# Learnin rate -> 0.01
# Random state -> 42
# Accuracy değeri -> 0,9551

predictList_xgbrfc_lda=xgbrfc_lda.predict(predict_test_lda)
print(predictList_xgbrfc_lda)
sonuc=donustur(predictList_xgbrfc_lda)
print(sonuc)
sonuc.to_csv("C:\\Users\\Dell\\Desktop\\XGBRFC predict list.csv",index=False)# Xgboost tahmin

# Parametreler

# test size oranı -> 0.2
# split random state -> 0
# Scaler -> Standart Scaler 
# Lda -> Kullanıldı, n=6
# Xgboost n sayısı -> 95
# Max dept -> 7
# Learnin rate -> 0.001
# Random state -> 7
# Accuracy değeri -> 0,9542

predictList_xgb=xgb.predict(xt_test)
print(predictList_xgb)
sonuc=donustur(predictList_xgb)
print(sonuc)
sonuc.to_csv("C:\\Users\\Dell\\Desktop\\XGBoost 2 predict list.csv",index=False)# Naive Bayes

# Parametreler

# test size oranı -> 0.2
# split random state -> 0
# Scaler -> Standart Scaler 
# Lda -> Kullanıldı, n=6
# Accuracy değeri -> 0,9450

predictList_gnb_lda=gnb_lda.predict(predict_test_lda)
print(predictList_gnb_lda)
sonuc=donustur(predictList_gnb_lda)
print(sonuc)
sonuc.to_csv("C:\\Users\\Dell\\Desktop\\Naive Bayes predict list.csv",index=False)#VotingClassifier tahmin

# Parametreler

# test size oranı -> 0.2
# split random state -> 0
# Scaler -> Standart Scaler
# Lda -> Kullanılmadı
# Logistic Regression -> random state=0
# RandomForestClassifier -> n=7, criterion = entropy, random state=14
# SVC -> kernel=linear
# XGBRFClassifier -> n=72, max_depth=9, learning rate=0.01, random state=42
# XGBClassifier -> n=100, max_depth=10, learning rate=0.01, random state=5
# VotingClassifier -> voting = hard
# Accuracy değeri -> 0,9551

predictList_vclf=vclf.predict(xt_test)
print(predictList_vclf)
sonuc=donustur(predictList_vclf)
print(sonuc)
sonuc.to_csv("C:\\Users\\Dell\\Desktop\\VotingClassifier predict list.csv",index=False)
# In[ ]:




