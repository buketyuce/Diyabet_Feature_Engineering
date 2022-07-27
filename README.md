# İş Problemi

Özellikleri belirtildiğinde kişilerin diyabet hastası olup olmadıklarını tahmin  edebilecek bir makine öğrenmesi modeli geliştirilmesi istenmektedir.

Modeli geliştirmeden önce gerekli olan veri analizi ve özellik mühendisliği adımlarını gerçekleştirmemiz gerekmekte.

# Veri Seti Hikayesi 

Veri seti ABD'deki Ulusal Diyabet-Sindirim-Böbrek Hastalıkları Enstitüleri'nde tutulan büyük veri setinin parçasıdır.

ABD'deki Arizona Eyaleti'nin en büyük 5. şehri olan Phoenix şehrinde yaşayan 21 yaş ve üzerinde olan Pima Indian kadınları üzerinde yapılan diyabet araştırması için kullanılan verilerdir.

Hedef değişken "outcome" olarak belirtilmiş olup; 1 diyabet test sonucunun pozitif oluşunu, 0 ise negatif oluşunu belirtmektedir.


# Değişkenler

Pregnancies: Hamilelik sayısı

Glucose: Oral glikoz tolerans testinde 2 saatlik plazma glikoz konsantrasyonu

Blood Pressure: Kan Basıncı (Küçük tansiyon) (mm Hg)

SkinThickness: Cilt Kalınlığı

Insulin: 2 saatlik serum insülini (mu U/ml)

DiabetesPedigreeFunction: Soydaki kişilere göre diyabet olma ihtimalini hesaplayan bir fonksiyon

BMI: Vücut kitle endeksi

Age: Yaş (yıl)

Outcome: Hastalığa sahip (1) ya da değil (0)


# Gerekli importları yapıp veriyi okutalım.

# Keşifçi Veri Analizi

![Ekran görüntüsü 2022-06-22 165838](https://user-images.githubusercontent.com/101973346/175047694-10f2456a-d025-4159-a815-898f8a06d6e4.png)

![Ekran görüntüsü 2022-06-22 165911](https://user-images.githubusercontent.com/101973346/175048073-aa97e04a-a071-48bd-a816-99395edf2be3.png)

![Ekran görüntüsü 2022-06-22 165931](https://user-images.githubusercontent.com/101973346/175048099-2cf7489f-f3c5-406c-afad-ca7385e43416.png)

![Ekran görüntüsü 2022-06-22 165952](https://user-images.githubusercontent.com/101973346/175048125-f82153c7-6cb9-4c92-8c98-d15a7fe27994.png)

![Ekran görüntüsü 2022-06-22 170010](https://user-images.githubusercontent.com/101973346/175048153-2830c78a-1d23-4a0e-852c-84da558877be.png)


# Numerik ve kategorik değişkenleri yakaladım.

![Ekran görüntüsü 2022-06-22 170159](https://user-images.githubusercontent.com/101973346/175048498-f3472bff-6a0b-4e68-91ea-2558a05f2628.png)

# Numerik ve kategorik değişkenlerin analizini yaptım.

![Ekran görüntüsü 2022-06-22 170159](https://user-images.githubusercontent.com/101973346/175049528-65111031-df44-4773-9270-67aa0972713a.png)
![Ekran görüntüsü 2022-06-22 170422](https://user-images.githubusercontent.com/101973346/175049549-95a133c1-3391-4f51-8362-fff81e48d756.png)
![Ekran görüntüsü 2022-06-22 170438](https://user-images.githubusercontent.com/101973346/175049564-7a94ef75-dff1-4564-a7a9-32bb2f587531.png)
![Ekran görüntüsü 2022-06-22 170451](https://user-images.githubusercontent.com/101973346/175049599-47943dc9-3080-4a32-b566-96b75309bf70.png)
![Ekran görüntüsü 2022-06-22 170504](https://user-images.githubusercontent.com/101973346/175049614-a4f74681-8fdb-42f3-9e16-7e2e0498f353.png)
![Ekran görüntüsü 2022-06-22 170518](https://user-images.githubusercontent.com/101973346/175049633-b2bebdf5-371e-4b0e-8cc3-6f22134f4504.png)
![Ekran görüntüsü 2022-06-22 170533](https://user-images.githubusercontent.com/101973346/175049652-7e4990ef-d907-4683-a968-25d26d8beea6.png)
![Ekran görüntüsü 2022-06-22 170549](https://user-images.githubusercontent.com/101973346/175049667-6ad82ebc-c280-48d2-99ae-3703f031fc4e.png)
![Ekran görüntüsü 2022-06-22 170603](https://user-images.githubusercontent.com/101973346/175049685-e54276eb-fa7d-417f-8fb7-0957982dbffb.png)

![Ekran görüntüsü 2022-06-22 170728](https://user-images.githubusercontent.com/101973346/175050001-57de2e7a-88a9-4ee0-84f4-928e7b36034c.png)

![Ekran görüntüsü 2022-06-22 170758](https://user-images.githubusercontent.com/101973346/175050021-366c26a8-00c4-4b13-8cdb-387ddd8ecbe8.png)

# Hedef değişken analizi yaptım.

![Ekran görüntüsü 2022-06-22 171000](https://user-images.githubusercontent.com/101973346/175050525-82fe4300-4035-49aa-84f2-849f3acc4154.png)
![Ekran görüntüsü 2022-06-22 171018](https://user-images.githubusercontent.com/101973346/175050546-d2da902a-2e84-4607-bd2e-e4b019baf5af.png)

# Korelasyon analizi yaptım.

![Ekran görüntüsü 2022-06-22 171110](https://user-images.githubusercontent.com/101973346/175050730-d4b3d84c-bc33-4d27-866e-3b961f022315.png)

# BASE MODEL KURULUMU

![Ekran görüntüsü 2022-06-22 171159](https://user-images.githubusercontent.com/101973346/175051137-1e7a399a-ed16-4c2e-99f8-9c53865e7a7c.png)

![Ekran görüntüsü 2022-06-22 171306](https://user-images.githubusercontent.com/101973346/175051163-a84f4bc8-29e3-4115-b13e-a80728f06d2d.png)

# Eksik değer analizi

# Aykırı gözlem analizi

## ÖZELLİK ÇIKARIMI

Yaş değişkenini katagorilere ayırıp yeni yaş değişkeni oluşturulması

BMI 18,5 asağısı underweight, 10.5 ile 24.9 arası normal, 24.9 ile 29.9 arası Overweight ve 30 üstü obese

Glukoz değerlerini kategorik değişkene çevirme

İnsülin değeri ile kategorik değişken türetme

![Ekran görüntüsü 2022-06-23 153323](https://user-images.githubusercontent.com/101973346/175299428-aa8f93a6-c39c-4c0f-aae0-83be8f6bb849.png)

## ENCODING

![Ekran görüntüsü 2022-06-23 153547](https://user-images.githubusercontent.com/101973346/175299877-ad9a023e-2465-453d-a121-ceee79f35de4.png)

![Ekran görüntüsü 2022-06-23 153607](https://user-images.githubusercontent.com/101973346/175299891-21da562b-9eb9-4299-bff6-406ec30a7159.png)

## STANDARTLAŞTIRMA

![Ekran görüntüsü 2022-06-23 153730](https://user-images.githubusercontent.com/101973346/175300147-cf26992a-f069-4498-8008-8ea90ecd2f39.png)

![Ekran görüntüsü 2022-06-23 153746](https://user-images.githubusercontent.com/101973346/175300161-3170f411-cca5-4633-99dd-3e12bf3387c5.png)

## MODELLEME

![Ekran görüntüsü 2022-06-23 153902](https://user-images.githubusercontent.com/101973346/175300389-13272dea-a012-4659-8cff-2cba3a3eff42.png)

## FEATURE İMPORTANCE

![Ekran görüntüsü 2022-06-23 153957](https://user-images.githubusercontent.com/101973346/175300607-35f0c770-41ab-468f-abb5-8c24c5ddbfb5.png)

![Ekran görüntüsü 2022-06-23 154021](https://user-images.githubusercontent.com/101973346/175300620-65480c34-6a8a-48b3-a7c6-b27ee4c04407.png)

