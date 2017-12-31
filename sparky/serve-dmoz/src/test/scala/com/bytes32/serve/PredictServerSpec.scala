package com.bytes32.serve

import com.twitter.finagle.http.Status._
import com.twitter.finatra.http.EmbeddedHttpServer
import com.twitter.inject.server.FeatureTest
import com.twitter.io.Buf

class PredictServerSpec extends FeatureTest {

  override val server = new EmbeddedHttpServer(new PredictServer)

  ignore("PredictServer#empty predict request gets empty predict response") {
    server.httpPost(
      path = "/predict",
      postBody =
        """
        {
          "texts": []
        }
        """,
      andExpect = Ok,
      withBody = """{"preds":[]}""")
  }

  ignore("PredictServer#technology predict a fashion category ") {
    server.httpPost(
      path = "/predict",
      postBody =
        """
        {
          "texts": ["ASOS | Online Shopping for the Latest Clothes & Fashion X ASOS Marketplace Home to the world's hottest new brands and vintage boutiques United Kingdom United Kingdom Change currency: £ GBP $ USD C$ CAD kr SEK kr NOK kr DKK ₣ CHF € EUR $ AUD ¥ RMB $ HKD $ NZD $ SGD NT$ TWD руб. RUB Other country sites United States › France › Deutschland › Italia › España › Australia › Россия › x Go Your Recent Search History Clear Welcome to ASOS Join | Sign In Hi , not you? Hi , sign out Home WOMEN Shop by Product SALE: DRESSES SALE: SHOES New In: Clothing New In: Shoes & Accs Outlet: Up To 70% Off Outlet: New In LOOPED: Sneaker Style Accessories Activewear ASOS Basic Tops Bags & Purses Beauty Coats & Jackets Curve & Plus Size Denim Designer Dresses Gifts Hoodies & Sweatshirts Jeans Jewellery & Watches Jumpers & Cardigans Jumpsuits & Playsuits Lingerie & Nightwear Loungewear Maternity Petite Shirts & Blouses Shoes Shorts Skirts Socks & Tights Sunglasses Swimwear & Beachwear T-Shirts & Vests Tall Tops Trousers & Leggings Workwear Suits Multipacks SAVE Brands adidas ASOS ASOS White Boohoo Chi Chi London Glamorous Miss Selfridge Missguided Monki New Look Nike Noisy May Office Reclaimed Vintage River Island Weekday A To Z Of Brands Exclusive To ASOS Eco Edit Shop ASOS Magazine Edits Occasionwear Holiday The Hotlist Wedding Shop Festival Street Classics Going Out-Out Workwear ASOS MARKETPLACE Up To 70% Off Sale! New In: Vintage New In: Independent Labels Marketplace Edits 90s Grunge Vintage Sportswear Festival MEN Shop by Product SALE: SHOES SALE: T-SHIRTS New In: Clothing New In: Shoes & Accs Outlet: New In Outlet: Up To 70% Off LOOPED: Sneaker Style Accessories Activewear Bags Blazers Caps & Hats Gifts Grooming Hoodies & Sweatshirts Jackets & Coats Jeans Jewellery Joggers Jumpers & Cardigans Loungewear Plus Size Polo Shirts Shirts Shoes, Boots & Trainers Shorts Suits Sunglasses Swimwear Tall Trousers & Chinos T-Shirts & Vests Underwear & Socks Watches Multi Packs SAVE Popular Brands Abercrombie & Fitch adidas Originals ASOS Cheap Monday Diesel Ellesse Fred Perry Jack & Jones New Balance New Look Nike Polo Ralph Lauren Reclaimed Vintage Religion River Island Selected Homme Vans A To Z Of Brands Shop by Edit Holiday Festival The Suit Guide New Trend Wedding Shop Sneaker Brands Jeans: New Styles Workwear ASOS MARKETPLACE Up To 70% Off Sale! New In: Vintage New In: Independent Labels Marketplace Edits 90s Grunge Vintage Sportswear Summer Essentials Help My Account Saved Items Bag   Removed from your bag... Your bag is empty VIEW SAVED ITEMS Qty x SAVE BAG TOTAL VIEW BAG PAY NOW Free DELIVERY Worldwide* *MORE INFO HERE  STUDENTS: 10% OFF 24/7 + MORE GOOD STUFF *Restrictions apply. Click banner for full terms. UNLIMITED NEXT-DAY DELIVERY TO THE UK ONLY £9.95 A YEAR *Restrictions apply. Click banner for full terms. FREE DELIVERY WORLDWIDE* *MORE INFO HERE THIS IS ASOS Your fashion and style destination SHOP WOMEN SHOP MEN FREE DELIVERY & RETURNS MORE INFO HERE STUDENTS: 10% OFF 24/7 + MORE GOOD STUFF NEXT-DAY DELIVERY ONLY £9.95 A YEAR Sign up for ASOS style news Questions? Help Track Order Delivery Returns What's in store Women Men Buy Gift Vouchers Follow Asos Facebook Twitter YouTube More About Asos Corporate Responsibility Jobs at ASOS Investors More Asos Sites Mobile and ASOS apps Marketplace Visit ASOS's international sites: United States France Deutschland Italia España Australia Россия Privacy & Cookies Terms & Conditions Accessibility About Us The celebrities named or featured on asos.com have not endorsed recommended or approved the items offered on site ©2017 asos.com Ltd All rights reserved Cookie Use ASOS uses cookies to ensure that we give you the best experience on our website. If you continue we assume that you consent to receive all cookies on all ASOS websites. More Info X"]
        }
        """,
      andExpect = Ok,
      withBody = """{"preds":[{"predictions":[7.609683E-13,3.7121806E-13,7.967529E-11,5.7463996E-14,4.222984E-11,4.9015183E-9,2.6509138E-8,2.3538577E-8,3.6340434E-11,2.8148678E-8,2.772427E-12,1.1120433E-7,0.99999964,6.361493E-8,5.721139E-13,7.723009E-8,3.9728146E-14,1.05649525E-13,4.4385007E-10,1.7390489E-8],"labels":["entertainment","news","finance","education","medical","sport","gardening","technology","music","cars","science","food","fashion","photography","health","bikes","property","parents","home","travel"],"max2":["fashion","food"]}]}""")
  }

  ignore("PredictServer#sports predict technology category from pcworld.co.uk") {
    val response = server.httpPost(
      path = "/predictWebSite",
      postBody =
        """
        {
          "uris": ["http://www.pcworld.co.uk"]
        }
        """,
      andExpect = Ok)

    assert(response.content.toUTF8String.contains(""","max2":["technology""""))
  }

  implicit class BufOps(val b:Buf) {
    def toUTF8String:String = {
      val arr = new Array[Byte](b.length)
      b.write(arr, 0)
      new String(arr, "UTF-8")
    }
  }

}
