package gr.ntua.ece.softeng18b.client.rest

import gr.ntua.ece.softeng18b.client.model.PriceInfo
import gr.ntua.ece.softeng18b.client.model.PriceInfoList
import gr.ntua.ece.softeng18b.client.model.Product
import gr.ntua.ece.softeng18b.client.model.ProductList
import gr.ntua.ece.softeng18b.client.model.Shop
import gr.ntua.ece.softeng18b.client.model.ShopList

class XmlRestCallResult implements RestCallResult {

    final def xml
    
    XmlRestCallResult(def xml) {
        this.xml = xml
    }

    @Override
    void writeTo(Writer w) {
        
    }

    @Override
    String getToken() { 
        null
    }

    @Override
    String getMessage() { 
        null
    }

    @Override
    ProductList getProductList() { 
        null
    }

    @Override
    Product getProduct() { 
        null
    }

    @Override
    ShopList getShopList() { 
        null
    }

    @Override
    Shop getShop() { 
        null
    }

    @Override
    PriceInfoList getPriceInfoList() {
        return null
    }
}

