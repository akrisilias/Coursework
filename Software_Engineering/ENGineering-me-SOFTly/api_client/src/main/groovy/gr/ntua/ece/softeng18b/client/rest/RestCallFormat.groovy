package gr.ntua.ece.softeng18b.client.rest

import groovy.json.JsonSlurper
import org.apache.http.HttpEntity

interface RestCallFormat {

    RestCallResult parse(HttpEntity entity) 
    
    String getName()
    
    static RestCallFormat JSON = new RestCallFormat(){
        @Override
        RestCallResult parse(HttpEntity entity)  {
            def json = new JsonSlurper().parse(entity.getContent(), "UTF-8")            
            return new JsonRestCallResult(json)
        }      
        
        @Override
        String getName() { "json" }
    }
    
    static RestCallFormat XML = new RestCallFormat() {
        @Override
        RestCallResult parse(HttpEntity entity) {
            def xml = new XmlSlurper().parse(entity.getContent())
            return new XmlRestCallResult(xml)
        }
        
        @Override
        String getName() { "xml" }
    }
    
}

