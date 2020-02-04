import { Component, OnInit } from '@angular/core';
import { DataService } from '../data.productsService';
import Map from 'ol/Map';
import Tile from 'ol/layer/Tile';
import OSM from 'ol/source/OSM';
import View from 'ol/View';
import { toLonLat } from 'ol/proj.js';
import { fromLonLat } from 'ol/proj.js';
import VectorSource from 'ol/source/Vector';
import {Icon, Style} from 'ol/style';
import Point from 'ol/geom/Point';
import Feature from 'ol/Feature';
import {Tile as TileLayer, Vector as VectorLayer} from 'ol/layer';

@Component({
  selector: 'app-shops',
  templateUrl: './shops.component.html',
  styleUrls: ['./shops.component.scss']
})

export class ShopsComponent implements OnInit {

  shops: any;
  Name: string;
  Address: string;
  Longtitude: string;
  Latitude: string;
  Withdrawn: string;
  Tags: string;
  map;
  vectorSource;
  feat;
  vectorLayer;
  ShopsLng: string[] = [];
  ShopsLat: string[] = [];
  i;

  view: number;
  status: string;
  sort: string;

  error:string = '';

  logged: boolean = false;


  constructor(private data: DataService) { }

  ngOnInit() {
	  this.status = 'ACTIVE';
    this.sort = 'id|DESC';

    if (this.data.auth != "") {
      this.logged = true;
    }
    else {
      this.logged = false;
    }

    this.data.getShops(this.status, this.sort)
      .subscribe(data => {
        this.shops = data
        console.log(this.shops);
		
	//	console.log(this.Longtitude);
	
		for (var i=0; i<(data as any).shops.length; i++) {
          this.ShopsLng.push((data as any).shops[i].lng)
          console.log(this.ShopsLng[i]);
		  this.ShopsLat.push((data as any).shops[i].lat)
          console.log(this.ShopsLat[i]);
        }
			console.log(this.ShopsLat.length);
			this.initilizeMap();
      }
    );

	  this.view = 20;
	  
	  

  }

  getClick() {
    this.data.getShops(this.status, this.sort)
      .subscribe(data => {
        this.shops = data
        console.log(this.shops);
        console.log("AEK");
        }
      );
  }

  postClick() {
    //this.data.firstClick();
    console.log('clicked');
    console.log(this.Name);

    this.data.postShops(this.Name, this.Address, this.Longtitude, this.Latitude, this.Withdrawn,this.Tags)
    .subscribe( data => {
      console.log(data);
      if ((data as any).message != null) {
        this.error = (data as any).message;
      }
      else {
        this.error = '';
      }
      this.getClick();
      }
    );

  }

  deleteClick(id:string) {
    console.log('delete clicked');
    console.log(id);
    this.data.deleteShop(id)
      .subscribe( data => {
        console.log(data);
        if ((data as any).message != null) {
          this.error = (data as any).message;
        }
        else {
          this.error = '';
        }
        this.getClick();
        }
      );
  }
  
  initilizeMap() {
	  
	var coord = fromLonLat([23.71622, 37.97945]);
	this.i=1;
	this.vectorSource = new VectorSource();
	console.log(this.ShopsLat.length);
	
	for (var i=0; i<this.ShopsLat.length; i++) {
		
		var [coord1, coord2]=fromLonLat([this.ShopsLng[i], this.ShopsLat[i]]);
		let feature = new Feature(
            new Point([coord1, coord2])
          );
		this.feat=feature;

          let style= new Style({
			image: new Icon(({
			color: 'yellow',
			crossOrigin: 'anonymous',
			src: 'assets/img/dot.png',
			imgSize: [20, 20]
			}))
		})

          // let value = parseFloat(datum.value);
          feature.setStyle(style);
          this.vectorSource.addFeature(feature);
		  
		  console.log("AEK");
		  
        }
	
		
    this.vectorLayer = new VectorLayer({
      source: this.vectorSource
    });

    this.map = new Map({
      target: 'map',
      layers: [
        new TileLayer({ source: new OSM() }),
        this.vectorLayer
      ],
      view: new View({
        center: coord,
        zoom: 10
      })
    });
	
	this.map.on('click', (evt) => {
		
		if (this.i==1)
			this.i=0;
		else
		   this.vectorSource.removeFeature(this.feat);
		
		
		//this.vectorSource.clear();
		var [coord1, coord2]=this.map.getEventCoordinate(evt.originalEvent);
		
		var lonlat = toLonLat([coord1, coord2]);
		var lon = lonlat[0];
		var lat = lonlat[1];
		this.Longtitude=lon;
		this.Latitude=lat;
		// …	
			console.log('clicked');
			
			
          this.feat = new Feature(
            new Point([coord1, coord2])
          );

          let style= new Style({
			image: new Icon(({
			color: 'red',
			crossOrigin: 'anonymous',
			src: 'assets/img/dot.png',
			imgSize: [20, 20]
			}))
		})

          // let value = parseFloat(datum.value);
          this.feat.setStyle(style);
          this.vectorSource.addFeature(this.feat);
	
	});

  }


}
