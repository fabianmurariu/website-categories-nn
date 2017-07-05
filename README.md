# Predict categories from text with pre-trainder word embeddings on dmoz dataset

## The plan

1. ~~get the data from [Harvard](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/OMV93V)~~
2. ~~use a crawler and an instance in the cloud to crawl 3+ million web page and store the HTML they return~~
3. ~~go through the categories tree and select a subset of categories to predict~~
4. ~~get word embeddings from [Standford](http://nlp.stanford.edu/data/glove.6B.zip)~~
5. ~~extact the text from the data and run Spark MLlib TF-IDF on the corpus~~
6. ~~For web-pages in english extend [this code](https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html) to train a NN predicting all the categories from 3~~
7. ~~Load the model in a JVM server and predict categories~~ 
8. Build a webservice that given a text will return predictions for categories

## The output from the webservice predicting pcworld as technology and asos as fashion
```text
===========================================================================
HTTP POST /predictWebSite
[Header]	Content-Length -> 101
[Header]	Content-Type -> application/json;charset=utf-8
[Header]	Host -> 127.0.0.1:57786
{
  "uris" : [
    "http://www.pcworld.co.uk",
    "http://www.asos.co.uk"
  ]
}
===========================================================================
14:56:19.914 [finagle/netty4-2] DEBUG com.bytes32.serve.PredictService - text for http://www.pcworld.co.uk is [PC World | Laptops, Tablets, iPads, Desktop PCs, Printers & More Search Menu Search Stores  Sign in Basket 0 Home Laptops Laptops MacBook Microsoft Surface Laptop accessories Mobile broadband Brand shops Buying for business? Learn more about Compare broadband deals Clearance Services Desktops Desktop PCs PC monitors Projectors Learn more about Buying for business? Brand shops Compare broadband deals Clearance Services iPad, Tablets & Mobile iPad Tablets Smart Tech Mobile Phones Broadband Learn more about eReaders Buying for business? Brand shops Compare broadband deals Clearance Services Software Software Games Learn more about Buying for business? Brand shop Clearance Services Printing Printers Paper Scanners Ink & Toner Office supplies Buying for business? Brand shop Learn more about Clearance Services Accessories iPad & tablet accessories Smart Tech PC Gaming Webcams PC speakers Laptop accessories Mice and keyboards Dictation Mobile phone accessories Buying for business? Brand shop Learn more about Clearance Network & Storage Networking Data storage Internal hard drives USB flash drives Smart Tech CD DVD and Blu-ray drives Blank media Mobile broadband Buying for business? Brand shops Learn more about Clearance Upgrades Components and upgrades Internal hard drives Raspberry Pi Clearance Buying for business? TV, Audio & Photo Televisions DVD, Blu-ray and home cinema Digital & Smart TV Sat nav Gaming Audio & headphones Cameras & Camcorders Photography accessories Buying for business? Brand shops Learn more about Compare digital TV packages Clearance Laptops Desktops iPad, Tablets & Mobile Software Printing Accessories Network & Storage Upgrades TV, Audio & Photo Shop deals as advertised on TV Shop deals in laptops Shop deals in desktops Shop deals in printers Shop deals in tv & entertainment Shop deals in accessories & smart tech As advertised on TV Sale Sale Sale Sale Sale Sale Sale Sale Sale Sale Sale Reasons to shop Like us Follow us Watch us Pin us Google+ Our blog Customer services Delivery & Recycling Track my Order Computer set up Recycling Information Returns & Cancellations Shopping with PC World Order online & collect in store Price Promise Gift Cards Payment & Credit Options Privacy & Cookies We're here to help Product Care Plans Customer Services Repairs Buying Guides Contact us Store finder Enter your postcode to find your nearest PC World store: Find stores Our other websites KNOWHOW | PC World Business | PC World Ireland | Currys Partmaster | Currys TechTalk | About DSG Retail Ltd Corporate site | Careers | PR & Media © DSG Retail Limited. DSG Retail Ltd, 1 Portal Way, London, W3 6RS, United Kingdom. Company registration number: 504877, VAT number: 226 6599 33 Terms & Conditions]
14:56:21.068 [finagle/netty4-2] DEBUG com.bytes32.serve.PredictService - text for http://www.asos.co.uk is [ASOS | Online Shopping for the Latest Clothes & Fashion X ASOS Marketplace Home to the world's hottest new brands and vintage boutiques United Kingdom United Kingdom Change currency: £ GBP $ USD C$ CAD kr SEK kr NOK kr DKK ₣ CHF € EUR $ AUD ¥ RMB $ HKD $ NZD $ SGD NT$ TWD руб. RUB Other country sites United States › France › Deutschland › Italia › España › Australia › Россия › x Go Your Recent Search History Clear Welcome to ASOS Join | Sign In Hi , not you? Hi , sign out Home WOMEN Shop by Product SALE: DRESSES SALE: SHOES New In: Clothing New In: Shoes & Accs Outlet: Up To 70% Off Outlet: New In LOOPED: Sneaker Style Accessories Activewear ASOS Basic Tops Bags & Purses Beauty Coats & Jackets Curve & Plus Size Denim Designer Dresses Gifts Hoodies & Sweatshirts Jeans Jewellery & Watches Jumpers & Cardigans Jumpsuits & Playsuits Lingerie & Nightwear Loungewear Maternity Petite Shirts & Blouses Shoes Shorts Skirts Socks & Tights Sunglasses Swimwear & Beachwear T-Shirts & Vests Tall Tops Trousers & Leggings Workwear Suits Multipacks SAVE Brands adidas ASOS ASOS White Boohoo Chi Chi London Glamorous Miss Selfridge Missguided Monki New Look Nike Noisy May Office Reclaimed Vintage River Island Weekday A To Z Of Brands Exclusive To ASOS Eco Edit Shop ASOS Magazine Edits Occasionwear Holiday The Hotlist Wedding Shop Festival Street Classics Going Out-Out Workwear ASOS MARKETPLACE Up To 70% Off Sale! New In: Vintage New In: Independent Labels Marketplace Edits 90s Grunge Vintage Sportswear Festival MEN Shop by Product SALE: SHOES SALE: T-SHIRTS New In: Clothing New In: Shoes & Accs Outlet: New In Outlet: Up To 70% Off LOOPED: Sneaker Style Accessories Activewear Bags Blazers Caps & Hats Gifts Grooming Hoodies & Sweatshirts Jackets & Coats Jeans Jewellery Joggers Jumpers & Cardigans Loungewear Plus Size Polo Shirts Shirts Shoes, Boots & Trainers Shorts Suits Sunglasses Swimwear Tall Trousers & Chinos T-Shirts & Vests Underwear & Socks Watches Multi Packs SAVE Popular Brands Abercrombie & Fitch adidas Originals ASOS Cheap Monday Diesel Ellesse Fred Perry Jack & Jones New Balance New Look Nike Polo Ralph Lauren Reclaimed Vintage Religion River Island Selected Homme Vans A To Z Of Brands Shop by Edit Holiday Festival The Suit Guide New Trend Wedding Shop Sneaker Brands Jeans: New Styles Workwear ASOS MARKETPLACE Up To 70% Off Sale! New In: Vintage New In: Independent Labels Marketplace Edits 90s Grunge Vintage Sportswear Summer Essentials Help My Account Saved Items Bag   Removed from your bag... Your bag is empty VIEW SAVED ITEMS Qty x SAVE BAG TOTAL VIEW BAG PAY NOW Free DELIVERY Worldwide* *MORE INFO HERE  STUDENTS: 10% OFF 24/7 + MORE GOOD STUFF *Restrictions apply. Click banner for full terms. UNLIMITED NEXT-DAY DELIVERY TO THE UK ONLY £9.95 A YEAR *Restrictions apply. Click banner for full terms. FREE DELIVERY WORLDWIDE* *MORE INFO HERE THIS IS ASOS Your fashion and style destination SHOP WOMEN SHOP MEN FREE DELIVERY & RETURNS MORE INFO HERE STUDENTS: 10% OFF 24/7 + MORE GOOD STUFF NEXT-DAY DELIVERY ONLY £9.95 A YEAR Sign up for ASOS style news Questions? Help Track Order Delivery Returns What's in store Women Men Buy Gift Vouchers Follow Asos Facebook Twitter YouTube More About Asos Corporate Responsibility Jobs at ASOS Investors More Asos Sites Mobile and ASOS apps Marketplace Visit ASOS's international sites: United States France Deutschland Italia España Australia Россия Privacy & Cookies Terms & Conditions Accessibility About Us The celebrities named or featured on asos.com have not endorsed recommended or approved the items offered on site ©2017 asos.com Ltd All rights reserved Cookie Use ASOS uses cookies to ensure that we give you the best experience on our website. If you continue we assume that you consent to receive all cookies on all ASOS websites. More Info X]
14:56:21.194 [Timer-0] INFO com.twitter.finatra.http.filters.AccessLoggingFilter - 127.0.0.1 - - [05/Jul/2017:14:56:21 +0000] "POST /predictWebSite HTTP/1.1" 200 1052 1520 "-"
---------------------------------------------------------------------------
[Status]	Status(200)
[Header]	Content-Type -> application/json; charset=utf-8
[Header]	Server -> Finatra
[Header]	Date -> Wed, 05 Jul 2017 14:56:20 GMT
[Header]	Content-Length -> 1052
{
  "preds" : [
    {
      "predictions" : [
        2.4127767E-17,
        3.6968116E-18,
        4.647949E-16,
        2.220812E-18,
        4.6302355E-23,
        7.523876E-23,
        4.5511077E-20,
        1.0,
        1.6818894E-20,
        1.9745627E-16,
        3.3073075E-19,
        3.0259375E-20,
        1.01233305E-13,
        6.40875E-11,
        8.78994E-28,
        5.3559624E-24,
        5.991734E-18,
        7.302525E-30,
        1.8646949E-13,
        9.0569984E-21
      ],
      "labels" : [ "entertainment", "news", "finance", "education", "medical", "sport", "gardening",
        "technology", "music", "cars", "science", "food", "fashion", "photography", "health",
        "bikes", "property", "parents", "home", "travel" ],
      "max2" : [ "technology", "photography" ]
    },
    {
      "predictions" : [
        7.609683E-13,
        3.7121806E-13,
        7.967529E-11,
        5.7463996E-14,
        4.222984E-11,
        4.9015183E-9,
        2.6509138E-8,
        2.3538577E-8,
        3.6340434E-11,
        2.8148678E-8,
        2.772427E-12,
        1.1120433E-7,
        0.99999964,
        6.361493E-8,
        5.721139E-13,
        7.723009E-8,
        3.9728146E-14,
        1.05649525E-13,
        4.4385007E-10,
        1.7390489E-8
      ],
      "labels" : [ "entertainment", "news", "finance", "education", "medical", "sport", "gardening",
        "technology", "music", "cars", "science", "food", "fashion", "photography", "health",
        "bikes", "property", "parents", "home", "travel" ],
      "max2" : [ "fashion", "food" ]
    }
  ]
}
```