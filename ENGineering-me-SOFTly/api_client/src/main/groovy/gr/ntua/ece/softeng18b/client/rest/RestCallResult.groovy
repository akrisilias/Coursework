package gr.ntua.ece.softeng18b.client.rest

import gr.ntua.ece.softeng18b.client.model.PriceInfo
import gr.ntua.ece.softeng18b.client.model.PriceInfoList
import gr.ntua.ece.softeng18b.client.model.Product
import gr.ntua.ece.softeng18b.client.model.ProductList
import gr.ntua.ece.softeng18b.client.model.Shop
import gr.ntua.ece.softeng18b.client.model.ShopList

interface RestCallResult {
    
    void writeTo(Writer w)
    
    String getToken()
    
    String getMessage()
    
    ProductList getProductList()
    
    Product getProduct()
    
    ShopList getShopList()
    
    Shop getShop()

    PriceInfoList getPriceInfoList()
    
}

