package gr.ntua.ece.softeng18b.client.model

import groovy.transform.Canonical 

@Canonical class Product {

    String id
    String name
    String description
    String category
    List<String> tags
    boolean withdrawn    
        
}

