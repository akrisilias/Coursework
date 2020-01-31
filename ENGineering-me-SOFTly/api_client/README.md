# softeng18b-rest-api-client
A Java/Groovy client for the Online Price Observatory REST API (part of [softeng18](https://github.com/saikos/softeng18b) project, used as an example in the Software Engineering course of the School of Electrical and Computer Engineering, National Technical University of Athens, 2018-2019).

The client is fully operational and demonstrates the use of:

* The [Apache Http Components fluent client API](https://hc.apache.org/) for performing HTTP requests.
* The [Wire Mock](http://wiremock.org/) library for mocking the responses of HTTP requests.
* The [Spock](http://spockframework.org/) framework for writing clean, declarative, maintainable test specifications.


The client can be embedded in any Java / Groovy application and it is developed for helping perform the functional / integration
tests of the Observatory back-end. See the test of [softeng18b](https://github.com/saikos/softeng18b/blob/master/src/test/groovy/gr/ntua/ece/softeng18b/RestAPISpecification.groovy).
