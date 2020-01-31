package gr.ntua.ece.softeng18b.client.rest

import groovy.json.JsonSlurper
import org.apache.http.HttpEntity
import org.apache.http.HttpResponse
import org.apache.http.client.ResponseHandler

class RestResponseHandler implements ResponseHandler<RestCallResult> {
    
    private final RestCallFormat format
    
    RestResponseHandler(RestCallFormat format) {
        this.format = format
    }       
    
    RestCallResult handleResponse(HttpResponse response) {
        HttpEntity entity = response.getEntity()
        def statusLine = response.getStatusLine()
        int code = statusLine.getStatusCode()
        if (code == 200) {            
            if (! format) throw new RuntimeException("No result handler")                        
            return format.parse(entity)            
        }
        else {
            throw new RuntimeException("Error $code: ${statusLine.getReasonPhrase()}")     
        }
    }                
}

