## EXNO-3-DS

## AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

## ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

## FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

## Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

## CODING AND OUTPUT:
```
NAME : YAMUNAASRI T S
REG NO : 212222240117
```
~~~
     import pandas as pd
     df=pd.read_csv("/content/Encoding Data.csv")
     df
~~~

![316296378-eca0e23d-6ea3-4685-ad3e-3a3c688afce4](https://github.com/aparnabalasubrmanian/EXNO-3-DS/assets/123351172/645b37b0-b212-4231-9fee-8642944fca48)



  ~~~
    from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
    pm=['Hot','Warm','Cold']
    e1=OrdinalEncoder(categories=[pm])
    e1.fit_transform(df[["ord_2"]])
~~~

![316296336-6f0fce59-7852-489d-a76f-db5988a45a3b](https://github.com/aparnabalasubrmanian/EXNO-3-DS/assets/123351172/12a7729e-b7b0-45a3-ae2f-fb034cda13e9)



~~~
    df['bo2']=e1.fit_transform(df[["ord_2"]])
    df
~~~

![316295056-84e9360b-9728-444d-bbe3-f7480e9633f6](https://github.com/aparnabalasubrmanian/EXNO-3-DS/assets/123351172/110739ad-4f70-4e76-9917-a46a124cceb8)

~~~
    df['bo2']=e1.fit_transform(df[["ord_2"]])
    df
~~~
![316295116-addbdb92-ff8a-41f3-af9e-bd97ac6800a2](https://github.com/aparnabalasubrmanian/EXNO-3-DS/assets/123351172/9bda3026-17ec-4a64-acae-6f5487ac208d)

~~~
    le=LabelEncoder()
    dfc=df.copy()
    dfc['ord_2']=le.fit_transform(dfc['ord_2'])
    dfc
~~~

![316295165-1c7de496-371e-4a21-a189-7ed70ecc2900](https://github.com/aparnabalasubrmanian/EXNO-3-DS/assets/123351172/b175b0d3-59d4-41da-b5ec-c321d0e5f32a)

  ~~~
    from sklearn.preprocessing import OneHotEncoder
    ohe=OneHotEncoder(sparse=False)
    df2=df.copy()
    enc=pd.DataFrame(ohe.fit_transform(df2[['nom_0']]))
    df2=pd.concat([df2,enc],axis=1)
    df2
~~~

![316295256-a8c5038b-2814-4b1d-8f85-c27b292c04d4](https://github.com/aparnabalasubrmanian/EXNO-3-DS/assets/123351172/b6f27d0b-7f28-43b1-bcaf-8570e7a15365)

  ~~~
    pd.get_dummies(df2,columns=["nom_0"])
~~~

![316295310-797cb3cf-cc31-4c39-ba12-af1d740bbbed](https://github.com/aparnabalasubrmanian/EXNO-3-DS/assets/123351172/b7e744d2-70f4-4eb7-aa3a-778fa0b56507)

  ~~~
    pip install --upgrade category_encoders
  ~~~

![316295375-eba5c171-3e23-483f-b4ed-6f4c47b5e89b](https://github.com/aparnabalasubrmanian/EXNO-3-DS/assets/123351172/84421e85-1d2f-45dd-8076-240d2666db75)

~~~
    from category_encoders import BinaryEncoder
    df=pd.read_csv("/content/data.csv")
    be=BinaryEncoder()
    nd=be.fit_transform(df['Ord_2'])
    fb=pd.concat([df,nd],axis=1)
    dfb1=df.copy()
    dfb
 
 ~~~

![316295450-9b983cb5-c712-49d8-bcd8-817ca7f56947](https://github.com/aparnabalasubrmanian/EXNO-3-DS/assets/123351172/eda26b77-8e16-40a6-bfc9-3f3230d6f8c5)

 ~~~
    from category_encoders import TargetEncoder
    te=TargetEncoder()
    cc=df.copy()
    new=te.fit_transform(X=cc["City"],y=cc["Target"])
    cc=pd.concat([cc,new],axis=1)
    cc
~~~

![316295534-0d2e7e51-9cb4-4530-a56d-39d1435d0249](https://github.com/aparnabalasubrmanian/EXNO-3-DS/assets/123351172/76af7fcc-aa44-4dfa-a62c-00901cae3d77)

~~~
    import pandas as pd
    from scipy import stats
    import numpy as np
    df=pd.read_csv("/content/Data_to_Transform.csv")
    df
~~~

![316295585-f71a4dd5-6389-4fb9-a209-81064b1d878f](https://github.com/aparnabalasubrmanian/EXNO-3-DS/assets/123351172/899386cd-c8f7-4197-b3dc-6a35a4180dfb)

~~~
    df.skew()
~~~

![316295623-3e30db2a-11cd-4980-a53d-87e495cdba2d](https://github.com/aparnabalasubrmanian/EXNO-3-DS/assets/123351172/c14d1924-1f6f-491f-b183-16c7c8045049)

~~~
    np.log(df["Highly Positive Skew"])
~~~

![316295660-ea34c360-81a5-4760-bc9e-2c6a9d07be6c](https://github.com/aparnabalasubrmanian/EXNO-3-DS/assets/123351172/ede24b90-c8bf-4487-a21c-9952312f6e47)

~~~
    np.reciprocal(df["Moderate Positive Skew"])
~~~

![316295703-790eadf6-e770-4874-943c-ddfb7bca5113](https://github.com/aparnabalasubrmanian/EXNO-3-DS/assets/123351172/35bed70d-3c0c-49a8-8ce8-d5f6f5a841a2)

~~~
    np.sqrt(df["Highly Positive Skew"])
~~~

![316295749-7e2ab48c-7321-477e-81e4-ba3d6b132bf5](https://github.com/aparnabalasubrmanian/EXNO-3-DS/assets/123351172/c91d7e8b-5bad-4b13-964b-f86f1e9ff9a7)

~~~
    np.square(df["Highly Positive Skew"])
~~~

![316295806-bcda767c-4193-4548-89ae-8ba49be8a323](https://github.com/aparnabalasubrmanian/EXNO-3-DS/assets/123351172/dc087c88-dc86-4096-8df3-9917ac787c8e)

~~~
   df["Highly Positive Skew_boxcox"],parameters=stats.boxcox(df["Highly Positive Skew"])
   df
~~~

![316295883-7632047a-430d-4b18-8bab-abc86a475a5a](https://github.com/aparnabalasubrmanian/EXNO-3-DS/assets/123351172/10e07c86-ecca-485e-a1e6-1c6598a47260)

~~~
    df["Moderate Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Moderate Negative Skew"])
    df.skew()
~~~

![316295945-90edf1a6-0480-49e3-9fe6-7ac276d2acf3](https://github.com/aparnabalasubrmanian/EXNO-3-DS/assets/123351172/0acb8caf-cfa4-4c51-b165-8970309d2b01)

~~~
    df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
    df.skew()

~~~
![316295988-af78ba87-bd09-43e8-a612-406cad9d7686](https://github.com/aparnabalasubrmanian/EXNO-3-DS/assets/123351172/f5996d8a-b22b-44bf-824d-98f16f610430)

~~~
   import matplotlib.pyplot as plt
   import seaborn as sns
   import statsmodels.api as sm
   import scipy.stats as stats

   sm.qqplot(df["Moderate Negative Skew"],line='45')

   plt.show()
  ~~~

![316296082-b72a5319-b492-4c91-981e-15bf8d38c539](https://github.com/aparnabalasubrmanian/EXNO-3-DS/assets/123351172/99b6477c-aa10-4e32-a392-1c866bb69c2c)

~~~
    sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
~~~

![316296150-9c011486-2fd5-4752-b434-702ecc5bebdc](https://github.com/aparnabalasubrmanian/EXNO-3-DS/assets/123351172/fc832ec2-75d2-47f0-bc02-ac842a21063b)

~~~
    from sklearn.preprocessing import QuantileTransformer
    qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)

    df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])

    sm.qqplot(df["Moderate Negative Skew"],line='45')
    plt.show()
    
~~~

![316296228-880ace95-7dd1-4732-941d-e007439a6fc5](https://github.com/aparnabalasubrmanian/EXNO-3-DS/assets/123351172/796de90e-461a-434b-bdf4-85a5506d0df3)


  ## RESULT:
  
  Hence performing Feature Encoding and Transformation process is Successful.
        

       
