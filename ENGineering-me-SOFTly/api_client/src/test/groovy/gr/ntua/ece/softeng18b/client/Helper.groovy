package gr.ntua.ece.softeng18b.client

import gr.ntua.ece.softeng18b.client.model.PriceInfo
import gr.ntua.ece.softeng18b.client.model.PriceInfoList
import gr.ntua.ece.softeng18b.client.model.Product
import gr.ntua.ece.softeng18b.client.model.ProductList
import gr.ntua.ece.softeng18b.client.model.Shop
import gr.ntua.ece.softeng18b.client.model.ShopList
import groovy.time.TimeCategory

import java.text.SimpleDateFormat

class Helper {

    private static final SimpleDateFormat DATE_FORMAT = new SimpleDateFormat("yyyy-MM-dd")
    
    static final String HOST  = "localhost"
    static final int PORT     = 8765          
    static final String TOKEN = "ABC123"
    static final String USER  = "user"
    static final String PASS  = "pass"
    
    static final Product newProduct(String id, Map productData) {
        Product p = new Product(productData)
        p.id = id
        return p
    }
    
    static final Shop newShop(String id, Map shopData) {
        Shop s = new Shop(shopData)
        s.id = id
        return s
    }
    
    //Place the fields in the same order as defined in the respective class    
    static final def PROD1_DATA = [
        name        : "FirstProduct",
        description : "FirstDescription",
        category    : "FirstCategory",
        tags        : ["Tags", "of", "first", "Product"],
        withdrawn   : false
    ]
    
    static final def PROD2_DATA = [
        name        : "SecondtProduct",
        description : "SecondDescription",
        category    : "SecondCategory",
        tags        : ["Tags", "of", "second", "Product"],
        withdrawn   : false
    ]
    
    static final def PROD3_DATA = [
        name        : "ProductName",
        description : "ProductDescription",
        category    : "FirstCategory",
        tags        : ["Tags", "of", "first", "Product"],
        withdrawn   : false
    ]
    
    static final Product PROD1    = newProduct("1", PROD1_DATA) 
    static final Product PROD2    = newProduct("2", PROD2_DATA) 
    static final Product PROD1UPD = newProduct("1", PROD3_DATA) 
    
    static final ProductList PRODUCTS = new ProductList(
        start   : 0,
        count   : 10,
        total   : 2,
        products: [PROD1, PROD2]
    )        
        
    //Place the fields in the same order as defined in the respective class    
    static final def SHOP1_DATA = [
        name     : "FistShop",
        address  : "AddressOfFirstShop",
        lat      : 37.97864720247794,
        lng      : 23.78350140530576,
        tags     : ["Tags", "of", "first", "shop"],
        withdrawn: false
    ]
    
    static final def SHOP2_DATA = [
        name     : "SecondShop",
        address  : "AddressOfSecondShop",
        lat      : 37.98136303504576,
        lng      : 23.78413117565094,
        tags     : ["Tags", "of", "second", "shop"],
        withdrawn: false
    ]
    
    static final def SHOP3_DATA = [
        name     : "OtherShop",
        address  : "OtherAddress",
        lat      : 37.97864720247794,
        lng      : 23.78350140530576,
        tags     : ["Tags", "of", "first", "shop"],
        withdrawn: false
    ]
    
    static final Shop SHOP1    = newShop("1", SHOP1_DATA)
    static final Shop SHOP2    = newShop("2", SHOP2_DATA)
    static final Shop SHOP1UPD = newShop("1", SHOP3_DATA)
    
    static final ShopList SHOPS = new ShopList(
        start: 0,
        count: 10,
        total: 2,
        shops: [SHOP1, SHOP2]
    )

    static final def PINFO_SUBMIT_DATA = [
        price: 13.25,
        dateFrom: "2019-02-21",
        dateTo: "2019-02-22",
        productId: PROD1.id,
        shopId: SHOP1.id
    ]

    static final PriceInfo PINFO1 = new PriceInfo(
        price: 13.25,
        date: "2019-02-21",
        productName: PROD1.name,
        productId: PROD1.id,
        productTags: PROD1.tags,
        shopId: SHOP1.id,
        shopName: SHOP1.name,
        shopTags: SHOP1.tags,
        shopAddress: SHOP1.address
    )

    static final PriceInfo PINFO2 = new PriceInfo(
        price: 13.25,
        date: "2019-02-22",
        productName: PROD1.name,
        productId: PROD1.id,
        productTags: PROD1.tags,
        shopId: SHOP1.id,
        shopName: SHOP1.name,
        shopTags: SHOP1.tags,
        shopAddress: SHOP1.address
    )

    static final PriceInfoList PINFO_LIST = new PriceInfoList(
        start: 0,
        count: 10,
        total: 2,
        prices: [PINFO1, PINFO2]
    )

    static int durationInDays(String dateFrom, String dateTo) {
        Date from = DATE_FORMAT.parse(dateFrom)
        Date to   = DATE_FORMAT.parse(dateTo)
        use(TimeCategory) {
            def duration = to - from
            return duration.days
        }
    }

    static List<String> elementsAt(List<String> source, List<Integer> indexes) {
        List<String> result = []
        int sz = indexes.size()
        for (int i = 0; i < sz; i++) {
            result.push(source[indexes[i]])
        }
        return result
    }

}

