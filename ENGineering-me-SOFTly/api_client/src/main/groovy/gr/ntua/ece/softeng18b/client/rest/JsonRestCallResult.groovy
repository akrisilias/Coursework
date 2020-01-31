package gr.ntua.ece.softeng18b.client.rest

import gr.ntua.ece.softeng18b.client.model.PriceInfo
import gr.ntua.ece.softeng18b.client.model.PriceInfoList
import gr.ntua.ece.softeng18b.client.model.Product
import gr.ntua.ece.softeng18b.client.model.Shop
import gr.ntua.ece.softeng18b.client.model.ProductList
import gr.ntua.ece.softeng18b.client.model.ShopList

class JsonRestCallResult implements RestCallResult {

    private final def json     
    
    JsonRestCallResult(def json) {
        this.json = json
    }

    @Override
    void writeTo(Writer w) {
        
    }

    @Override
    String getToken() {
        return json['token'] as String
    }

    @Override
    String getMessage() {
        return json['message'] as String
    }

    @Override
    ProductList getProductList() {        
        int start    = json['start'] as Integer
        int count    = json['count'] as Integer
        long total   = json['total'] as Long        
        def products = json['products']
        List<Product> productList = products.collect { p ->
            parseProduct(p)
        }

        return new ProductList(
            start   : start,
            count   : count,
            total   : total,
            products: productList
        )
    }

    @Override
    Product getProduct() {
        return parseProduct(json)
    }

    protected Product parseProduct(p) {
        return new Product(
            id         : p['id'] as String,
            name       : p['name'] as String,
            description: p['description'] as String,
            category   : p['category'] as String,
            tags       : p['tags'] as List,
            withdrawn  : p['withdrawn'] as Boolean
        )
    }

    @Override
    ShopList getShopList() {
        int start  = json['start'] as Integer
        int count  = json['count'] as Integer
        long total = json['total'] as Long        
        def shops  = json['shops']
        List<Shop> shopList = shops.collect { s ->
            parseShop(s)
        }

        return new ShopList(
            start : start,
            count : count,
            total : total,
            shops : shopList
        )
    }

    protected Shop parseShop(s) {
        return new Shop(
            id        : s['id'] as String,
            name      : s['name'] as String,
            address   : s['address'] as String,
            lat       : s['lat'] as double,
            lng       : s['lng'] as double,            
            tags      : s['tags'] as List,
            withdrawn : s['withdrawn'] as Boolean
        )
    }

    @Override
    Shop getShop() {
        return parseShop(json)
    }

    @Override
    PriceInfoList getPriceInfoList() {
        return parsePrices(json)
    }

    protected PriceInfoList parsePrices(json) {
        int start = json['start'] as Integer
        int count = json['count'] as Integer
        long total = json['total'] as Long
        def prices = json['prices']
        List<PriceInfo> priceList = prices.collect { p ->
            parsePriceInfo(p)
        }

        return new PriceInfoList(
            start : start,
            count : count,
            total : total,
            prices: priceList
        )
    }

    protected PriceInfo parsePriceInfo(p) {
        return new PriceInfo(
            price: p['price'] as Double,
            date: p['date'] as String,
            productName: p['productName'] as String,
            productId: p['productId'] as String,
            productTags: p['productTags'] as List<String>,
            shopId: p['shopId'] as String,
            shopName: p['shopName'] as String,
            shopTags: p['shopTags'] as List<String>,
            shopAddress: p['shopAddress'] as String,
            shopDist: p['shopDist'] as Integer
        )
    }
}

